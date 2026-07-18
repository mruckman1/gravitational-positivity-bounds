"""Phase-2a solver stack: certified gravitational positivity bounds (D=7).

Certifies statements of the form (units M=1, 8piG=1):
    alpha_G * 8piG + alpha_2 * g_2 + alpha_3 * g_3  >=  0
i.e.  g_2 + (alpha_3/alpha_2) g_3 >= -(alpha_G/alpha_2) * 8piG/M^2,
for ANY UV-completable scalar+gravity EFT in D=7 whose spectrum has spins
J <= j_audit (explicit scope, CMRS-grade statements are the j->inf limit).

Pipeline (roles strictly separated — the recon soundness ledger):
  1. CANDIDATE SHAPING (unsound-by-construction, never trusted): an LP over
     smearing coefficients f(p) = sum_n a_n p^n (integer n, odd D) with
     numeric-quadrature constraint rows on an (m, J) grid plus the Bessel
     impact-parameter closure fhat(b) >= 0 on a b-grid. scipy/HiGHS.
  2. EXACT COLUMNS: alpha_G = sum a_n/(n-1), alpha_2 = 2 sum a_n/(n+1),
     alpha_3 = sum a_n/(n+3) — exact rationals after rationalizing a_n.
  3. CERTIFICATION (the only thing reported): for the rationalized f, the
     per-spin heavy action E_J(m) = int_0^1 f(p) C2imp[m,J](-p^2) dp is
     computed in CLOSED FORM (sympy; rational + atan/log — the combined
     integrand is regular at p=m so no (m^2-p^2) denominators survive) and
     proven >= 0 on ALL m in [1, inf) by certified interval arithmetic
     (mpmath.iv, outward rounding) with adaptive bisection in w = 1/m,
     plus an exact w->0 leading-order check. Audit failure at any (J, m*)
     feeds the point back to the LP grid (refinement loop); certification
     is only ever claimed from a PASSING audit.
3-way discipline: certified-with-scope / audit-failure-refine / hard error.
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import linprog

from qgse.verifiers.gravity_positivity import (c2imp_kernel, _M2, _U, _P,
                                               run_assertion_battery)

_W = sp.Symbol("w", positive=True)

# sympy's heuristic GCD raises HeuristicGCDFailed on high-J null-smear
# kernels, and the sparse-ring path calls it UNCONDITIONALLY (the
# USE_HEU_GCD config gate only covers the dense path). Patch the ring-level
# GCD with a deterministic dense-path fallback.
from sympy.polys import polyconfig as _polyconfig
from sympy.polys import rings as _rings
from sympy.polys.heuristicgcd import heugcd as _heugcd
from sympy.polys.polyerrors import HeuristicGCDFailed as _HGF
_polyconfig.setup("USE_HEU_GCD", False)
_orig_gcd_ZZ = _rings.PolyElement._gcd_ZZ


def _gcd_ZZ_safe(f, g):
    try:
        return _heugcd(f, g)
    except _HGF:
        R = f.ring
        Pf = sp.Poly(f.as_expr(), *R.symbols, domain=R.domain)
        Pg = sp.Poly(g.as_expr(), *R.symbols, domain=R.domain)
        h = Pf.gcd(Pg)
        return (R.from_expr(h.as_expr()),
                R.from_expr(Pf.quo(h).as_expr()),
                R.from_expr(Pg.quo(h).as_expr()))


_rings.PolyElement._gcd_ZZ = _gcd_ZZ_safe


# -- kernels at u = -p^2, numeric layer ----------------------------------------- #
def _linprog_retry(*args, tries=5, **kwargs):
    """HiGHS intermittently returns 'Status 0: Not Set' — a transient
    non-completion on borderline dense LPs. It is nondeterministic ACROSS
    runs (the identical problem that fails here solved cleanly during the
    harvest certifications), so retrying the SAME call lets a flaky
    non-completion recover instead of aborting a multi-hour certification.
    Result-preserving: a HiGHS solve that RUNS TO COMPLETION on identical
    input is deterministic, so a successful retry returns the identical
    optimum — byte-identical to a first-try success. On persistent failure
    the last (failed) result is returned so the caller raises exactly as
    before."""
    res = None
    for _ in range(max(1, tries)):
        res = linprog(*args, **kwargs)
        if getattr(res, "success", False):
            return res
    return res


@functools.lru_cache(maxsize=None)
def _kernel_expr(J: int, D: int):
    """C2imp[m^2, J](-p^2), sympy expr in (_M2, _P).

    Pure function of the integer keys (J, D): the returned sympy expr is
    immutable and every caller only reads it (.subs / lambdify / multiply,
    all non-mutating), so memoizing is result-preserving and byte-identical
    to recomputing. This kernel build (sp.series/cancel) is the dominant
    pipeline cost; without the cache it is rebuilt once PER POWER inside
    _columns and again in both rows_grid and audit."""
    return sp.cancel(sp.together(c2imp_kernel(J, D, u=-_P**2)))


def _kernel_fn(J: int, D: int):
    """Fast float callable (m2, p) for LP rows."""
    return sp.lambdify((_M2, _P), _kernel_expr(J, D), "numpy")


def _bessel_transform_fn(n: int, D: int, pmax: float = 1.0):
    """t_n(b) = Gamma((D-2)/2) int_0^P p^n J_nu(bp)/(bp/2)^nu dp,
    nu=(D-4)/2. The m->inf closure column (numeric; shaping only)."""
    nu = (D - 4) / 2.0
    gam = float(mp.gamma((D - 2) / 2.0))

    def t(b: float) -> float:
        if b < 1e-9:
            return gam / float(mp.gamma((D - 2) / 2.0)) \
                * pmax**(n + 1) / (n + 1)
        f = lambda p: p**n * float(mp.besselj(nu, b * p)) / (b * p / 2.0)**nu
        return gam * float(mp.quad(f, [0, pmax]))
    return t




_UP = sp.Symbol("up")


@functools.lru_cache(maxsize=None)
def _xk_kernel(k: int, J: int, D: int):
    """u-smeared crossing null X_{k,u}[m^2,J] (CMRS eq 51), validated:
    X_k[J=0] == 0 identically; regular at u = -m^2; X_2 identical to the
    independent eq-45 construction (ratio 1 symbolically); forward limit
    proportional to the eq-27 Taylor-lattice null rows."""
    from qgse.verifiers.gravity_positivity import pj_poly
    m2, u, up = _M2, _U, _UP
    pj_u = pj_poly(J, D, 1 + 2 * u / m2)
    t1 = ((2 * m2 + u) / (u * m2 * (m2 + u))
          * m2 * pj_u / (u * m2 * (m2 + u))**(k // 2))
    pj_up = pj_poly(J, D, 1 + 2 * up / m2)
    integrand = ((2 * m2 + up) * (m2 - up) * (m2 + 2 * up)
                 / (m2 * (u - up) * up * (m2 - u) * (m2 + up)
                    * (m2 + u + up))
                 * m2 * pj_up / (up * m2 * (m2 + up))**(k // 2))
    ser = sp.series(sp.together(integrand) * up**(k // 2 + 2), up, 0,
                    k // 2 + 3).removeO()
    res = sp.expand(ser).coeff(up, k // 2 + 1)
    return sp.cancel(sp.together(t1 - res))


@functools.lru_cache(maxsize=None)
def _xk_expr(k: int, J: int, D: int):
    """X_{k}[m^2,J](-p^2) as expr in (_M2, _P). Pure in (k, J, D); cached
    (see _kernel_expr) — the hot inner kernel, rebuilt once per power
    without the cache."""
    return sp.cancel(sp.together(_xk_kernel(k, J, D).subs(_U, -_P**2)))


# -- spec ------------------------------------------------------------------------ #
@dataclass
class GravRaySpec:
    """Certify one ray: fix alpha_3/alpha_2 = ray_alpha, minimize alpha_G."""
    D: int = 7
    ray_alpha: float = -1.0 / 3.0          # published D=7 ray slopes
    powers: tuple = (2, 3, 4, 5, 6, 7)     # f(p) basis exponents (odd D)
    # null-smear columns {k: powers of p} — REQUIRED for feasibility at
    # production j_max (measured: the C2imp-only basis is INFEASIBLE at
    # J<=42); zero low-side cost (exact nulls). CMRS Fig-4 space.
    x_smears: tuple = ((4, (0, 1, 2, 3, 4, 5)), (6, (0, 1, 2, 3)))
    j_max: int = 42                        # LP grid spins
    j_audit: int = 60                      # exact-audit spin scope
    p_max: object = 1                      # extended-u endpoint (Rational;
    #                                        >1 adds the meromorphy axiom)
    delta_base: float = 1e-3               # reserve-blend starting delta
    n_xgrid: int = 400                     # m^2 = 1/(1-x) grid
    b_grid: tuple = (0.25, 80.0, 320)      # (lo, hi, count)
    max_refine: int = 8

    def validate(self):
        if self.D == 4:
            raise ValueError("D=4 gravity REFUSED without an IR-regulator "
                             "flag (b-space graviton measure diverges, CMRS "
                             "eq 70) — use D >= 5")
        if self.D % 2 == 0:
            raise ValueError("Phase-2a implements odd D only (integer "
                             "powers => fully certified audit); even-D "
                             "half-integer bases are Phase-2b")
        if any(n < 2 for n in self.powers):
            raise ValueError("powers must be >= 2 (alpha_G convergence)")


class GravityRayVerifier:
    """LP + certified-audit pipeline for one gravitational ray bound."""

    def __init__(self):
        fails = [n for n, ok in run_assertion_battery() if not ok]
        if fails:
            raise RuntimeError(f"kernel battery FAILED: {fails} — refusing "
                               "to build a solver on unvalidated kernels")

    # -- LP --------------------------------------------------------------- #
    def _lp_rows(self, spec: GravRaySpec, extra_pts):
        ms, rows = [], []
        xs = np.linspace(0.0, 1.0 - 1.0 / spec.n_xgrid, spec.n_xgrid)
        m2s = 1.0 / (1.0 - xs)
        kfns = {J: _kernel_fn(J, spec.D)
                for J in range(0, spec.j_max + 1, 2)}
        xfns = {(k, J): sp.lambdify((_M2, _P), _xk_expr(k, J, spec.D),
                                    "numpy")
                for (k, _pw) in spec.x_smears
                for J in range(0, spec.j_max + 1, 2)}
        pmf = float(sp.Rational(spec.p_max))
        pgrid = np.linspace(1e-4, pmf, max(240, int(240 * pmf)))

        def full_row(J, m2, kf):
            vals = kf(m2, pgrid)
            row = [np.trapezoid(pgrid**n * vals, pgrid)
                   for n in spec.powers]
            for (k, pws) in spec.x_smears:
                xv = xfns[(k, J)](m2, pgrid)
                row += [np.trapezoid(pgrid**i * xv, pgrid) for i in pws]
            arr = np.asarray(row, dtype=float)
            if not np.all(np.isfinite(arr)):
                # float overflow at high J x large m2 (inf/inf -> nan in
                # lambdified rationals): recompute in mpmath
                exprs = ([_P**n * _kernel_expr(J, spec.D)
                          for n in spec.powers]
                         + [_P**i * _xk_expr(k, J, spec.D)
                            for (k, pws) in spec.x_smears for i in pws])
                arr = np.array([float(mp.quad(sp.lambdify(
                    _P, e.subs(_M2, m2), "mpmath"), [0, 1]))
                    for e in exprs])
            # row-wise max-normalization: positive rescaling of a >= 0
            # constraint is sound and kills the dynamic-range problem
            mx = np.abs(arr).max()
            return list(arr / mx) if mx > 0 else list(arr)

        for J, kf in kfns.items():
            for m2 in m2s:
                rows.append(full_row(J, float(m2), kf))
                ms.append((J, float(m2)))
        for (J, m2) in extra_pts:
            # audit-failure feedback rows: adaptive quadrature (1e-15), not
            # trapezoid — quadrature error here caused refinement STALLS
            # (LP satisfied its approximate row while the exact audit kept
            # failing by a hair at the same point)
            exprs = ([_P**n * _kernel_expr(J, spec.D) for n in spec.powers]
                     + [_P**i * _xk_expr(k, J, spec.D)
                        for (k, pws) in spec.x_smears for i in pws])
            row = []
            for e in exprs:
                g = sp.lambdify(_P, e.subs(_M2, m2), "mpmath")
                row.append(float(mp.quad(g, [0, 1])))
            rows.append(row)
            ms.append(("extra", J, m2))
        # Bessel closure rows (m -> inf continuum)
        lo, hi, cnt = spec.b_grid
        tfns = [_bessel_transform_fn(n, spec.D, pmf) for n in spec.powers]
        n_x = sum(len(pws) for (_k, pws) in spec.x_smears)
        for b in np.linspace(lo, hi, int(cnt)):
            rows.append([t(float(b)) for t in tfns] + [0.0] * n_x)
            ms.append(("bessel", float(b)))
        return np.array(rows), ms

    def _solve_lp(self, spec: GravRaySpec, extra_pts):
        rows, tags = self._lp_rows(spec, extra_pts)
        n_x = sum(len(pws) for (_k, pws) in spec.x_smears)
        n = len(spec.powers) + n_x
        zx = [0.0] * n_x                        # nulls: zero low side
        Pf = float(sp.Rational(spec.p_max))
        c = np.array([Pf**(p - 1) / (p - 1)
                      for p in spec.powers] + zx)                 # alpha_G
        a2 = np.array([2.0 * Pf**(p + 1) / (p + 1)
                       for p in spec.powers] + zx)
        a3 = np.array([Pf**(p + 3) / (p + 3) for p in spec.powers] + zx)
        A_eq = np.vstack([a2, a3 - spec.ray_alpha * a2])
        b_eq = np.array([1.0, 0.0])
        scale0 = np.abs(rows).sum(axis=1) + 1e-12
        b_pos = np.array([-1e-6 * sc if t[0] == "extra" else 0.0
                          for t, sc in zip(tags, scale0)])
        res = _linprog_retry(c, A_ub=-rows, b_ub=b_pos,
                             A_eq=A_eq, b_eq=b_eq,
                             bounds=[(None, None)] * n, method="highs")
        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")
        opt = float(res.fun)
        # ---- certifiability-first two-stage solve: the UNCAPPED optimum
        # needs |a| ~ 1e4 cancelling coefficients (uncertifiable: audit
        # bounds scale with |R1|). Stage A: re-minimize alpha_G under
        # |a_n| <= A_CAP (weaker but tame bound). Stage B: maximize the
        # minimum row slack subject to alpha_G <= 1.01 * capped optimum.
        # v1 trades tightness for certificates; the null-smear basis (v2)
        # buys tightness back. ----
        A_CAP = 1000.0
        resA = _linprog_retry(c, A_ub=-rows, b_ub=b_pos,
                              A_eq=A_eq, b_eq=b_eq,
                              bounds=[(-A_CAP, A_CAP)] * n, method="highs")
        if not resA.success:
            return res.x, opt
        optA = float(resA.fun)
        scale = np.abs(rows).sum(axis=1) + 1e-12
        nrows = rows / scale[:, None]
        c2 = np.zeros(n + 1); c2[-1] = -1.0
        # slack ONLY on (m, J) rows: Bessel-closure profiles touch zero at
        # oscillation minima by NATURE ((1-cos b)/b^2 archetype), so uniform
        # slack across them is structurally impossible — they remain plain
        # feasibility rows. The certified statement is J-scoped; margins
        # matter exactly where the audit checks.
        slack_ind = np.array([[0.0] if t[0] == "bessel" else [1.0]
                              for t in tags])
        b_ub2_base = np.array([-1e-6 * sc / sc if t[0] == "extra" else 0.0
                               for t, sc in zip(tags, scale)])
        A_ub2 = np.hstack([-nrows, slack_ind])
        b_ub2 = b_ub2_base
        gA = np.append(c, 0.0)
        A_ub2 = np.vstack([A_ub2, gA])
        b_ub2 = np.append(b_ub2, optA + 0.15 * abs(optA) + 1e-9)
        A_eq2 = np.hstack([A_eq, np.zeros((2, 1))])
        # prefer small coefficients inside the slack stage (audit error
        # terms scale with |R1| ~ coefficient size): try shrinking caps
        res2 = None
        for cap2 in (300.0, A_CAP):
            r2 = _linprog_retry(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2,
                                b_eq=b_eq,
                                bounds=[(-cap2, cap2)] * n + [(0, None)],
                                method="highs")
            if r2.success and r2.x[-1] > 1e-9:
                res2 = r2
                break
        if res2 is None:
            res2 = r2
        if res2.success and res2.x[-1] > 1e-9:
            aG = float(gA[:n] @ res2.x[:n])
            print(f"    [robust stage] capped opt {optA:.4f} -> certified-"
                  f"candidate alpha_G {aG:.4f}, min-slack "
                  f"{res2.x[-1]:.3e}, max|a| {np.abs(res2.x[:n]).max():.1f}",
                  flush=True)
            return res2.x[:n], aG
        return resA.x, optA

    # -- exact audit -------------------------------------------------------- #
    @staticmethod
    def _rationalize(a, max_den=10**5):
        return [sp.Rational(Fraction(float(v)).limit_denominator(max_den))
                for v in a]

    @staticmethod
    def _exact_smear_integral(integrand_expr, D: int, P=sp.Integer(1)):
        """EXACT int_0^1 <integrand> dp for integrands of the class
        N(p; m2) / (c(m2) * (m2 + p^2)^b) — covers f*C2imp + null smears
        (all kernels validated regular at the p=m corner, so only
        (m2+p^2)-powers survive). sympy's symbolic-parameter definite
        integrate is UNSOUND here (verified wrong result); instead:
        successive exact division to Q(p) + sum_j S_j(p)/(m2+p^2)^j
        (deg S_j <= 1), then closed forms:
          int p/(m2+p^2)^j : rational (log((m2+1)/m2)/2 at j=1),
          int 1/(m2+p^2)^j : recursion down to atan(1/sqrt(m2))/sqrt(m2).
        Returns (Eq_rational, r_atan, r_log, const) with
        E = (Eq + r_atan*atan(1/sqrt(m2))/sqrt(m2) + r_log*log((m2+1)/m2)/2)
            / const."""
        p, m2 = _P, _M2
        num, den = sp.fraction(sp.cancel(sp.together(integrand_expr)))
        core = m2 + p**2
        b = 0
        rest = den
        while True:
            q, r = sp.div(sp.Poly(rest, p), sp.Poly(core, p))
            if r.is_zero:
                rest = q.as_expr()
                b += 1
            else:
                break
        const = rest                        # p-free by construction
        if sp.Poly(const, p).degree() != 0:
            raise RuntimeError(f"unexpected denominator {sp.factor(den)}")
        # N/(core^b): peel S_j layers
        S = {}
        cur = sp.Poly(num, p)
        for j in range(b, 0, -1):
            cur, rem = sp.div(cur, sp.Poly(core, p))
            S[j] = rem                       # deg <= 1, over (m2+p^2)^j
        Q = cur                              # polynomial part
        Eq = sum(cc * P**(sp.Integer(mono[0]) + 1)
                 / (sp.Integer(mono[0]) + 1)
                 for mono, cc in zip(Q.monoms(), Q.coeffs())) \
            if not Q.is_zero else sp.Integer(0)
        # closed forms: A_j = int_0^1 dp/(m2+p^2)^j = rational + at_j * ATAN
        #               B_j = int_0^1 p dp/(m2+p^2)^j = rational (+LOG at j=1)
        # ATAN := atan(1/sqrt(m2))/sqrt(m2), LOG := log((m2+1)/m2)/2
        A_rat, A_at = {0: None}, {}
        A_rat[1], A_at[1] = sp.Integer(0), sp.Integer(1)
        for j in range(2, b + 1):
            # A_j = [p/(2(j-1) m2 (m2+p^2)^{j-1})]_0^P + (2j-3)/(2(j-1)m2) A_{j-1}
            bd = P / (2 * (j - 1) * m2 * (m2 + P**2)**(j - 1))
            fac = sp.Rational(2 * j - 3, 2 * (j - 1)) / m2
            A_rat[j] = bd + fac * A_rat[j - 1]
            A_at[j] = fac * A_at[j - 1]
        B_rat, B_lg = {}, {}
        B_rat[1], B_lg[1] = sp.Integer(0), sp.Integer(1)
        for j in range(2, b + 1):
            B_rat[j] = (1 / m2**(j - 1) - 1 / (m2 + P**2)**(j - 1)) \
                / (2 * (j - 1))
            B_lg[j] = sp.Integer(0)
        r_atan, r_log = sp.Integer(0), sp.Integer(0)
        for j, rem in S.items():
            cs = rem.all_coeffs() if not rem.is_zero else [0]
            s1 = cs[0] if len(cs) == 2 else 0
            s0 = cs[-1] if cs else 0
            if s0 != 0:
                Eq += s0 * A_rat[j]
                r_atan += s0 * A_at[j]
            if s1 != 0:
                Eq += s1 * B_rat[j]
                r_log += s1 * B_lg[j]
        return Eq, r_atan, r_log, const

    def _audit_spin(self, spec: GravRaySpec, a_rat, J: int):
        """Certified positivity of E_J(m) = int f * C2imp on m in [1, inf),
        i.e. w = 1/m in (0, 1]. Two regimes:
          small w (0, W_SW]: EXACT — atan/log replaced by alternating-series
            Taylor sandwiches with rigorous remainder terms bounded via
            monomial-wise absolute values (valid for w > 0), reducing to
            rational-polynomial nonnegativity decided by Sturm. This is
            where interval arithmetic dies of the dependency problem
            (R0 and R1*atan each vary ~1e24 while E ~ 1e-24) and exact
            cancellation is the only sound tool.
          large w [W_SW, 1]: certified interval bisection (mpmath.iv) —
            no deep cancellation there.
        Returns (ok, why)."""
        p = _P
        f = sum(a_rat[i] * p**n for i, n in enumerate(spec.powers))
        integrand = f * _kernel_expr(J, spec.D)
        ix = len(spec.powers)
        for (k, pws) in spec.x_smears:
            xk = _xk_expr(k, J, spec.D)
            h = sum(a_rat[ix + j] * p**i for j, i in enumerate(pws))
            integrand += h * xk
            ix += len(pws)
        Eq, r0, r1, const = self._exact_smear_integral(
            integrand, spec.D, P=sp.Rational(spec.p_max))
        w = _W
        sub = {_M2: 1 / w**2}
        R0 = sp.cancel(sp.together(Eq.subs(sub) / const.subs(sub)
                                   if Eq != 0 else sp.Integer(0)))
        R1 = sp.cancel(sp.together((r0 * w).subs(sub) / const.subs(sub))) \
            if r0 != 0 else sp.Integer(0)      # multiplies atan(w)
        R2 = sp.cancel(sp.together(r1.subs(sub) / (2 * const.subs(sub)))) \
            if r1 != 0 else sp.Integer(0)      # multiplies log(1+w^2)
        return self._audit_wpieces(R0, R1, R2, J,
                                   P=sp.Rational(spec.p_max))

    @staticmethod
    def _prescreen_negative(R0, R1, R2, P, W_SW, J):
        """Refuse-ONLY numerical probe of E(w)=R0+R1 atan(Pw)+R2 log(1+P^2w^2)
        on (0,1]. Returns a refusal message if E is CLEARLY negative somewhere
        (so the expensive exact passes can be skipped), else None. NEVER
        certifies: a 'looks positive' result falls through to the exact audit
        unchanged, and a certifiable functional (E>=0) is never flagged, so
        the SET of certifiable designs is unchanged -- only genuine refusals
        are made fast. Working precision is scaled to the coefficient
        magnitude so a positive-with-margin E is never falsely flagged. Opt-in
        (see fast_refuse gate in _audit_wpieces): it can reorder the failing
        spin a refinement loop feeds back on, so it is OFF by default to keep
        champion certificates byte-identical, and ON for campaigns (where any
        sound certificate is an equally valid data point). Any evaluation
        failure => None (fall through)."""
        import math as _math
        w = _W
        try:
            Pf = mp.mpf(int(P.p)) / int(P.q)
            hi = mp.mpf(int(W_SW.p)) / int(W_SW.q)
            mid = sp.Rational(1, 3)

            def _mag(R):
                if R == 0:
                    return 0.0
                try:
                    return abs(float(R.subs(w, mid)))
                except Exception:
                    return 0.0
            M = max(_mag(R0), _mag(R1), _mag(R2), 1.0)
            old = mp.mp.dps
            mp.mp.dps = max(30, 26 + int(_math.log10(M + 1.0)))
            try:
                z = lambda x: mp.mpf(0)
                f0 = sp.lambdify(w, R0, "mpmath") if R0 != 0 else z
                f1 = sp.lambdify(w, R1, "mpmath") if R1 != 0 else z
                f2 = sp.lambdify(w, R2, "mpmath") if R2 != 0 else z

                def _E(wv):
                    return (f0(wv) + f1(wv) * mp.atan(Pf * wv)
                            + f2(wv) * mp.log(1 + (Pf * wv)**2))
                n = 800
                worst, wworst = mp.mpf("1e40"), None
                for i in range(1, n + 1):
                    wv = mp.mpf(i) / n                     # (0, 1]
                    E = _E(wv)
                    if E < worst:
                        worst, wworst = E, wv
                step = mp.mpf(1) / n
                for _ in range(24):                        # refine around min
                    step /= 2
                    for cand in (wworst - step, wworst + step):
                        if 0 < cand <= 1:
                            E = _E(cand)
                            if E < worst:
                                worst, wworst = E, cand
                if worst < mp.mpf("-1e-8"):
                    if wworst <= hi:
                        return (f"J={J}: small-w exact sandwich unprovable "
                                f"at K=26")
                    return (f"J={J}: large-w segment unprovable at "
                            f"w~{float(wworst):.5f} (m ~ {1/float(wworst):.4f})")
                return None
            finally:
                mp.mp.dps = old
        except Exception:
            return None

    def _audit_wpieces(self, R0, R1, R2, J, P=sp.Integer(1)):
        """Two-regime certified positivity proof for
        E(w) = R0 + R1 atan(P w) + R2 log(1+P^2 w^2) on w in (0, 1].
        Shared by the D=7 ray and D=10 susy verifiers; endpoint P is the
        extended-u smearing endpoint (exact Rational)."""
        w = _W
        W_SW = sp.Rational(1, 2) / P

        # opt-in refuse-only fast path (campaigns): skip the expensive exact
        # passes on a clearly-negative E. Never blocks a certifiable design;
        # OFF by default so champion certificates stay byte-identical.
        if getattr(self, "fast_refuse", False):
            _pre = self._prescreen_negative(R0, R1, R2, P, W_SW, J)
            if _pre is not None:
                return False, _pre

        # ---- small-w regime: exact sandwich + Sturm (series in P*w) ----
        # K-escalation is monotonically SOUND: the alternating-series sandwich
        # is a valid lower bound at every K, so a larger K only converts
        # refusals into proofs, never a pass into anything else (passing
        # audits break at the same K as before -- byte-identical results).
        # K=36 added 2026-07-16 after give-sweep refusals at K=26 with
        # sub-1e-3 reserve slack (floor_sweep experiment).
        ok_small = False
        for K in (12, 18, 26, 36):
            S_at = sum((-1)**i * (P * w)**(2 * i + 1) / (2 * i + 1)
                       for i in range(K))
            S_lg = sum((-1)**(i + 1) * (P * w)**(2 * i) / i
                       for i in range(1, K + 1))
            rem_at = (P * w)**(2 * K + 1) / (2 * K + 1)
            rem_lg = (P * w)**(2 * K + 2) / (K + 1)
            LB = R0 + R1 * S_at + R2 * S_lg \
                - _abs_bound(R1) * rem_at - _abs_bound(R2) * rem_lg
            num, den = sp.fraction(sp.cancel(sp.together(LB)))
            # denominator: positive on (0,1]? (powers of w, 1+w^2 factors)
            if not _poly_pos_on_interval(sp.Poly(sp.expand(den), w), W_SW):
                num, den = -num, -den
                if not _poly_pos_on_interval(sp.Poly(sp.expand(den), w),
                                             W_SW):
                    continue
            npoly = sp.Poly(sp.expand(num), w)
            cs = list(reversed(npoly.all_coeffs()))
            if _bernstein_nonneg(cs, sp.Rational(0), W_SW) or \
                    _poly_nonneg_on_interval(npoly, W_SW):
                ok_small = True
                break
        if not ok_small:
            return False, f"J={J}: small-w exact sandwich unprovable at K=36"

        # ---- large-w regime [1/2, 1]: segment-wise rational atan/log
        # enclosures + exact Sturm. Interval bisection PROVEN unusable here
        # (profiled: 3.36M boxes — the dependency problem at coefficient
        # scale). Per segment [a,b]: atan(w) in [At(a), At(b)] (monotone),
        # enclosed by outward-rounded rationals; E_lb(w) = R0 + R1*At_mid
        # - |R1|*At_half (same for log) is rational -> Sturm on [a,b].
        # Adaptive splitting on failure; the robust-LP margin makes ~1e2
        # segments suffice.
        def _enclose(fun, x_lo, x_hi):
            mp.mp.dps = 30
            lo, hi = fun(mp.mpf(x_lo)), fun(mp.mpf(x_hi))
            from fractions import Fraction as _F
            # 12-digit outward rounding: Sturm cost scales superlinearly in
            # coefficient bit-length; 30-digit enclosures were pure waste
            pad = sp.Rational(1, 10**10)
            rlo = sp.Rational(_F(str(mp.nstr(lo, 14)))) - pad
            rhi = sp.Rational(_F(str(mp.nstr(hi, 14)))) + pad
            return (rlo + rhi) / 2, (rhi - rlo) / 2

        absR1 = _abs_bound(R1) if R1 != 0 else sp.Integer(0)
        absR2 = _abs_bound(R2) if R2 != 0 else sp.Integer(0)

        def seg_prove(a_, b_, depth=0):
            if depth > 22:
                return False, (float((a_ + b_) / 2), None)
            w0 = (a_ + b_) / 2
            h = (b_ - a_) / 2
            # FIRST-ORDER Taylor enclosures around the midpoint: the slopes
            # are EXACT rationals (atan' = 1/(1+w0^2), log' = 2w0/(1+w0^2)),
            # errors quadratic in h (|atan''| <= 2/3, |log''| <= 2 on [0,1])
            # — constant enclosures (linear error) needed depth ~21 at
            # near-tangent spins; quadratic needs ~10.
            at_mid, at_delta = _enclose_pt(mp.atan, P * w0)
            lg_mid, lg_delta = _enclose_pt(lambda t: mp.log(1 + t * t),
                                           P * w0)
            d_at = P / (1 + (P * w0)**2)
            d_lg = 2 * P**2 * w0 / (1 + (P * w0)**2)
            err_at = at_delta + h**2 * P**2 * sp.Rational(1, 3)
            err_lg = lg_delta + h**2 * P**2
            LBs = (R0
                   + R1 * (at_mid + d_at * (w - w0)) - absR1 * err_at
                   + R2 * (lg_mid + d_lg * (w - w0)) - absR2 * err_lg)
            num_, den_ = sp.fraction(sp.cancel(sp.together(LBs)))
            pn, pd = sp.Poly(sp.expand(num_), w), sp.Poly(sp.expand(den_), w)
            csn = list(reversed(pn.all_coeffs()))
            csd = list(reversed(pd.all_coeffs()))
            if _bernstein_nonneg(csd, a_, b_, max_depth=8) and \
                    _bernstein_nonneg(csn, a_, b_):
                return True, None
            if _bernstein_nonneg([-x for x in csd], a_, b_, max_depth=8) \
                    and _bernstein_nonneg([-x for x in csn], a_, b_):
                return True, None
            mid_ = (a_ + b_) / 2
            ok1, bad1 = seg_prove(a_, mid_, depth + 1)
            if not ok1:
                return False, bad1
            return seg_prove(mid_, b_, depth + 1)

        n_seg = 24
        span = 1 - W_SW
        for i in range(n_seg):
            a_ = W_SW + span * sp.Rational(i, n_seg)
            b_ = W_SW + span * sp.Rational(i + 1, n_seg)
            ok, bad = seg_prove(a_, b_)
            if not ok:
                wmid = bad[0]
                return False, (f"J={J}: large-w segment unprovable at "
                               f"w~{wmid:.5f} (m ~ {1/wmid:.4f})")
        return True, None

    # -- reserve: the max-margin functional of the whole basis ---------------- #
    def _reserve(self, spec: GravRaySpec):
        """Chebyshev-center LP: maximize the minimum normalized (m,J)-row
        slack subject only to alpha_2 = 1 (no ray constraint). Blending
        delta of this into any ray functional buys uniform audit margin at
        O(delta) bound cost — decoupling tightness from auditability (no
        simple positive profile works: the J=2 threshold row forces the
        sign-mixed structure only the LP finds)."""
        rows, tags = self._lp_rows(spec, [])
        n_x = sum(len(pws) for (_k, pws) in spec.x_smears)
        n = len(spec.powers) + n_x
        zx = [0.0] * n_x
        a2 = np.array([2.0 / (p + 1) for p in spec.powers] + zx)
        scale = np.abs(rows).sum(axis=1) + 1e-12
        nrows = rows / scale[:, None]
        sl = np.array([[0.0] if t[0] == "bessel" else [1.0] for t in tags])
        c2 = np.zeros(n + 1); c2[-1] = -1.0
        res = _linprog_retry(c2, A_ub=np.hstack([-nrows, sl]),
                             b_ub=np.zeros(len(nrows)),
                             A_eq=np.hstack([a2, [0.0]]).reshape(1, -1),
                             b_eq=np.array([1.0]),
                             bounds=[(-300.0, 300.0)] * n + [(0, None)],
                             method="highs")
        if not res.success or res.x[-1] <= 1e-6:
            raise RuntimeError("no reserve functional with positive margin")
        return res.x[:n], float(res.x[-1])

    # -- pipeline ------------------------------------------------------------ #
    def certify_ray(self, spec: GravRaySpec, log=print) -> dict:
        spec.validate()
        reserve, r_slack = self._reserve(spec)
        log(f"[reserve] max-margin functional: slack {r_slack:.3e}")
        extra = []
        for it in range(spec.max_refine):
            t0 = time.time()
            a, obj = self._solve_lp(spec, extra)
            log(f"[iter {it}] LP alpha_G = {obj:.6f} ({time.time()-t0:.0f}s)")
            delta = spec.delta_base * (4 ** min(it, 4))  # reserve blend
            a = np.asarray(a) + delta * np.asarray(reserve)
            a_rat = self._rationalize(a)
            # exact columns, renormalized to alpha_2 = 1 exactly
            Pq = sp.Rational(spec.p_max)
            a2 = sum(2 * q * Pq**(n + 1) / (n + 1)
                     for q, n in zip(a_rat[:len(spec.powers)], spec.powers))
            if a2 <= 0:
                raise RuntimeError("alpha_2 <= 0 after rationalization")
            a_rat = [q / a2 for q in a_rat]
            aG = sum(q * Pq**(n - 1) / (n - 1)
                     for q, n in zip(a_rat[:len(spec.powers)], spec.powers))
            a3 = sum(q * Pq**(n + 3) / (n + 3)
                     for q, n in zip(a_rat[:len(spec.powers)], spec.powers))
            failures = []
            t0 = time.time()
            for J in range(0, spec.j_audit + 1, 2):
                ok, why = self._audit_spin(spec, a_rat, J)
                if not ok:
                    failures.append((J, why))
                    log(f"    audit FAIL {why}")
                    if len(failures) >= 3:
                        break
            log(f"[iter {it}] audit: {'PASS' if not failures else 'FAIL'} "
                f"to J<={spec.j_audit} ({time.time()-t0:.0f}s)")
            if not failures:
                stmt = (f"g_2 + ({sp.nsimplify(a3)}) g_3 M^2 + "
                        f"({sp.nsimplify(aG)}) 8piG/M^2 >= 0")
                return {"D": spec.D, "ray_alpha_requested": spec.ray_alpha,
                        "ray_alpha_certified": float(a3),
                        "c_cert": float(aG), "c_cert_exact": str(aG),
                        "alpha3_exact": str(a3),
                        "functional": [str(q) for q in a_rat],
                        "powers": list(spec.powers),
                        "statement": stmt,
                        "scope": ((f"EXTENDED-u DOMAIN p_max="
                                   f"{sp.Rational(spec.p_max)} (AXIOM: "
                                   "meromorphy+discreteness, CMRS App. B); "
                                   if sp.Rational(spec.p_max) > 1 else "")
                                  + f"UV-completable scalar+gravity EFTs in "
                                  f"D={spec.D} with all spins J <= "
                                  f"{spec.j_audit} (exact audit; Bessel "
                                  "closure imposed in shaping)"),
                        "refine_iterations": it}
            # feed audit failures back into the LP grid
            for (J, why) in failures:
                if isinstance(why, str) and "m ~" in why:
                    m_star = float(why.split("m ~")[1].split(")")[0])
                    # dips SLIDE along a continuum under single-point pins
                    # (observed ~1%/iteration walk): pin a SPREAD so the
                    # whole sliding path is caught at once
                    for fac in (0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10):
                        extra.append((J, (m_star * fac)**2))
                else:
                    extra += [(J, 1.0 + 0.01 * k) for k in range(1, 6)]
        raise RuntimeError(f"no certifiable functional after "
                           f"{spec.max_refine} refinement iterations")

# -- exact-audit helpers --------------------------------------------------------- #
def _abs_bound(R):
    """Rational function bound: |R(w)| <= _abs_bound(R)(w) for w > 0, via
    monomial-wise absolute values of numerator and a positive denominator
    (denominators here are powers of w and (1+w^2)-type: positive)."""
    if R == 0:
        return sp.Integer(0)
    num, den = sp.fraction(sp.cancel(sp.together(R)))
    w = _W
    nb = sum(abs(c) * w**int(k[0])
             for k, c in zip(sp.Poly(sp.expand(num), w).monoms(),
                             sp.Poly(sp.expand(num), w).coeffs()))
    db = sp.Poly(sp.expand(den), w)
    if all(c > 0 for c in db.coeffs()):
        return nb / den
    if all(c < 0 for c in db.coeffs()):
        return nb / (-den)
    # fall back: lower-bound positive denominator via its smallest
    # coefficient structure is unsafe — refuse (caller tries higher K)
    raise ValueError("denominator sign not manifest")


def _poly_nonneg_on_interval(pl: sp.Poly, hi) -> bool:
    """Exact: pl(w) >= 0 on (0, hi] over QQ (Sturm/sqf)."""
    if pl.is_zero:
        return True
    if pl.eval(hi) < 0 or pl.eval(hi / 2) < 0:
        return False
    _, sqf = pl.sqf_list()
    for q, e in sqf:
        if e % 2 == 0:
            continue
        n = q.count_roots(0, hi)
        if q.eval(sp.Integer(0)) == 0:
            n -= 1
        if n > 0:
            # odd-multiplicity root inside (0, hi]: sign change possible;
            # accept only if it is exactly at hi with pl >= 0 just below
            if q.eval(hi) == 0 and q.count_roots(0, hi - hi / 1000) == \
                    (1 if q.eval(sp.Integer(0)) == 0 else 0):
                continue
            return False
    # no sign change in (0, hi): sign = sign near 0+ = sign of lowest coeff
    lead_low = None
    for k in range(pl.degree() + 1):
        c = pl.nth(k)
        if c != 0:
            lead_low = c
            break
    return lead_low is None or lead_low > 0


def _poly_pos_on_interval(pl: sp.Poly, hi) -> bool:
    """Exact: pl(w) > 0 on (0, hi]."""
    if pl.is_zero:
        return False
    if pl.eval(hi) <= 0:
        return False
    n = pl.count_roots(0, hi)
    if pl.eval(sp.Integer(0)) == 0:
        n -= 1
    if n > 0:
        return False
    for k in range(pl.degree() + 1):
        c = pl.nth(k)
        if c != 0:
            return c > 0
    return False

def _poly_pos_by_intervals(pl: sp.Poly, hi) -> bool:
    """Fast rigorous positivity of an exact-rational-coefficient polynomial
    on (0, hi]: adaptive-precision Horner interval bisection. Precision is
    scaled to the cancellation depth (coeff scale / value scale ~ w^-k), so
    tiny-w boxes are decided at high dps instead of infinite subdivision.
    Returns False when inconclusive (caller falls back to exact Sturm)."""
    iv = mp.iv
    coeffs = [sp.Rational(c) for c in pl.all_coeffs()]
    if not coeffs:
        return False
    # lowest-order nonzero coefficient must be positive (w->0+ sign)
    for c in reversed(coeffs):
        if c != 0:
            if c < 0:
                return False
            break

    def horner(wbox, dps):
        iv.dps = dps
        out = iv.mpf(0)
        for c in coeffs:
            out = out * wbox + iv.mpf(int(c.p)) / iv.mpf(int(c.q))
        return out

    import math

    def prove(lo, hi_, depth=0):
        # precision scaled to cancellation depth at w ~ lo (value ~ w^4 vs
        # coefficient scale): 8 digits per decade of smallness + headroom
        dps = min(max(30 + int(8 * -math.log10(max(float(lo), 1e-30))), 30),
                  400)
        try:
            val = horner(iv.mpf([lo, hi_]), dps)
        except Exception:
            return False
        if val.a > 0:
            return True
        if depth > 48:
            return False
        mid = (lo + hi_) / 2
        return prove(lo, mid, depth + 1) and prove(mid, hi_, depth + 1)

    return prove(mp.mpf(10) ** -30, mp.mpf(str(float(hi))), 0)

def _poly_nonneg_on_seg(pl: sp.Poly, a, b) -> bool:
    """Exact: pl >= 0 on [a, b] over QQ."""
    if pl.is_zero:
        return True
    if pl.eval(a) < 0 or pl.eval(b) < 0:
        return False
    _, sqf = pl.sqf_list()
    for q, e in sqf:
        if e % 2 == 1 and q.count_roots(a, b) - \
                (1 if q.eval(a) == 0 else 0) - \
                (1 if q.eval(b) == 0 else 0) > 0:
            return False
    return True


def _poly_pos_on_seg(pl: sp.Poly, a, b) -> bool:
    if pl.is_zero:
        return False
    if pl.eval(a) <= 0 or pl.eval(b) <= 0:
        return False
    return pl.count_roots(a, b) - (1 if pl.eval(a) == 0 else 0) - \
        (1 if pl.eval(b) == 0 else 0) == 0

def _enclose_pt(fun, x0):
    """Tight rational enclosure of fun(x0): value +- delta (outward)."""
    mp.mp.dps = 30
    v = fun(mp.mpf(str(float(x0))))
    from fractions import Fraction as _F
    mid = sp.Rational(_F(str(mp.nstr(v, 14))))
    return mid, sp.Rational(1, 10**10)


def _bernstein_nonneg(coeffs, a, b, depth=0, max_depth=26):
    """Certified: polynomial with exact rational `coeffs` (ascending, in w)
    is >= 0 on [a, b]. Bernstein-basis positivity: after affine map to
    [0,1], if all Bernstein coefficients >= 0 the polynomial is >= 0 on the
    box (sufficient); subdivide at the midpoint on mixed signs. Pure exact
    rational arithmetic, no GCD/Sturm — O(deg^2) per box."""
    n = len(coeffs) - 1
    if n < 0:
        return True
    # affine map w = a + (b-a) t, t in [0,1]: powers via synthetic shifts
    # p(t) = sum c_k (a + (b-a) t)^k — compute monomial coeffs in t
    ba = b - a
    mono = [sp.Rational(0)] * (n + 1)
    apow = [sp.Rational(1)]
    for _ in range(n):
        apow.append(apow[-1] * a)
    from math import comb
    for k, ck in enumerate(coeffs):
        if ck == 0:
            continue
        bapow = sp.Rational(1)
        for j in range(k + 1):
            mono[j] += ck * comb(k, j) * apow[k - j] * bapow
            bapow *= ba
    # monomial -> Bernstein: B_i = sum_{j<=i} mono[j] C(i,j)/C(n,j)
    bern = []
    neg = pos = False
    for i in range(n + 1):
        bi = sum(mono[j] * comb(i, j) / comb(n, j) for j in range(i + 1))
        bern.append(bi)
        if bi < 0:
            neg = True
        elif bi > 0:
            pos = True
    if not neg:
        return True
    # endpoint values are exact Bernstein endpoints: genuine negativity?
    if bern[0] < 0 or bern[-1] < 0:
        return False
    if depth >= max_depth:
        return False
    mid = (a + b) / 2
    return (_bernstein_nonneg(coeffs, a, mid, depth + 1, max_depth) and
            _bernstein_nonneg(coeffs, mid, b, depth + 1, max_depth))
