"""4D EFT positivity-bounds verifier — dispersive sum rules / EFT-hedron.

Phase 5 of the program: the same certified-SDP architecture as the bootstrap
verifiers, pointed at 4D physics. Setup (Caron-Huot & Van Duong 2011.02957,
"CHVD"; validated against Adams et al. hep-th/0602178 and cross-checked with
AHH 2012.15849): single real massless scalar in d=4, 2->2 amplitude M(s,t),
crossing symmetric, s+t+u=0, all states at m^2 >= M^2 (gap), twice-subtracted
fixed-t dispersion (axioms: analyticity + Regge |M/s^2| -> 0). Low-energy
couplings (CHVD eq 2.3): M_low = -g^2[1/s+1/t+1/u] - lambda
+ g_2(s^2+t^2+u^2) + g_3(stu) + g_4(s^2+t^2+u^2)^2 + ...

Moments over the positive heavy average <.> (CHVD eq 2.15, d=4:
<F> = sum_{J even} 16 pi (2J+1) int_{M^2}^inf dm^2/(pi m^2) rho_J(m^2) F):
  g_2 = <1/m^4>,  g_3 = <(3 - 2X)/m^6>,  g_4 = <1/(2 m^8)>,  X := J(J+1),
plus crossing NULL constraints <n_k> = 0:
  n_4 = 2X(X-8)/m^8,   n_5 = 2X(2X^2 - 43X + 150)/m^10.

EVERY row used by the SDP is derived symbolically at import time from the
master dispersion kernel
  K = (2m^2+t) P_J(1+2t/m^2) / ((m^2-s)(m^2+s+t)(m^2+t))
and cross-checked against the closed forms above — NOT hand-typed. Reason:
the published eq (2.16/18) kernel is inconsistent with eqs (2.15)+(2.22) by
one power of m^2 (found and resolved symbolically during recon); deriving and
asserting is the only trap-proof path. P_J(1+y) enters via the exact Taylor
formula P_J(1+y) = sum_r y^r/(2^r r!^2) prod_{i=1..r}(X - i(i-1)) so rows are
exact polynomials in X.

SOUNDNESS (the load-bearing point, opposite of the modular s_max case):
truncating spins J <= J_max DELETES dual constraints, so an SDPB "bound" can
be SPURIOUS — demonstrated by explicit counterexample during recon (J_max=2
+ Regge block "certifies" gt_3 >= -10.179 while a 2-state spectrum sits at
-10.612). Therefore NO bound is certified here on SDPB's word alone:
  (1) the SDP imposes J = 0..J_max (even) PLUS the exact Regge block
      q(xi) = lim_{x->inf} z.rows(x, X=xi(1+x))/(1+x)^{k_max-2} >= 0 (lossless);
  (2) the returned functional is rounded to rationals and re-verified in
      EXACT arithmetic: per-spin Sturm positivity on x in [0,inf) for all
      even J <= j_audit, and — where implemented (deg_X = 2, i.e. the
      n4-only problem) — a COMPLETE tail proof for all J via
      disc_X < 0 beyond x0 plus a Cauchy spin bound on the compact strip;
  (3) problems whose tail proof is not yet implemented are certified with an
      EXPLICIT SCOPE "spectra with all spins J <= j_audit" (never silently).
Certified output states which of (2)/(3) applies. 3-way discipline as
everywhere in qgse: optimal-with-audit / scoped / hard error — never a
plausible number without a category.

Units: M = 1 throughout (bounds are on gt_k = g_k M^{2k-4}/g_2, dimensionless).
Substitution m^2 = 1 + x, x >= 0; rows are numerators after clearing
(1+x)^{k_max}. Only rho_J >= 0 is used (homogeneous problem: ratios only).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sympy as sp
import mpmath as mp

from qgse.verifiers.bootstrap import _ensure_env, _in_dir

_X, _XI, _S, _T, _M2 = sp.symbols("X xi s t m2", positive=True)
_x = sp.Symbol("x", nonnegative=True)


# -- exact row generation from the master kernel ------------------------------ #
def _pj_taylor(y, X, order):
    """P_J(1+y) = sum_r y^r/(2^r r!^2) prod_{i=1..r}(X - i(i-1)), X = J(J+1).
    Exact in X; truncated at y^order (sufficient for total degree `order`)."""
    out = sp.Integer(0)
    for r in range(order + 1):
        prod = sp.Integer(1)
        for i in range(1, r + 1):
            prod *= (X - i * (i - 1))
        out += y**r * prod / (sp.Integer(2)**r * sp.factorial(r)**2)
    return out


def _kernel_coeffs(deg: int):
    """Taylor coefficients f[a,b] of s^a t^b of the master kernel
    K = (2m^2+t) P_J(1+2t/m^2) / ((m^2-s)(m^2+s+t)(m^2+t)),
    exact polynomials in X over 1/m^2 powers. Verified against CHVD eq (2.22):
    g_2 = f[0,0]/2, g_3 = -f[0,1], g_4 = f[2,0]/4 = f[0,2]/8 + null."""
    pj = _pj_taylor(2 * _T / _M2, _X, deg)
    # NOTE: this is m^2 x the eq-(2.16/18) kernel AS PRINTED — the printed one
    # is inconsistent with eqs (2.15)+(2.22) by one power of m^2 (recon find).
    # Sanity anchor: at s=t=0 this kernel = 2/m^4, so g_2 = f00/2 = <1/m^4>.
    K = (2 * _M2 + _T) * pj / ((_M2 - _S) * (_M2 + _S + _T) * (_M2 + _T))
    Kser = sp.expand(sp.series(sp.series(K, _S, 0, deg + 1).removeO(),
                               _T, 0, deg + 1).removeO())
    f = {}
    for a in range(deg + 1):
        for b in range(deg + 1 - a):
            f[(a, b)] = sp.expand(Kser.coeff(_S, a).coeff(_T, b))
    return f


def _derived_rows():
    """Derive (g2, g3, n4, n5) as exact functions of (X, 1/m^2) from the kernel
    and ASSERT the closed forms. Returns dict name -> (expr_in_X_over_m2, k)
    where k = the 1/m^2 power (row scales as 1/m^(2k))."""
    f = _kernel_coeffs(5)
    g2 = sp.together(f[(0, 0)] / 2)
    g3 = sp.together(-f[(0, 1)])
    g4s = sp.together(f[(2, 0)] / 4)
    g4t = sp.together(f[(0, 2)] / 8)
    n4 = sp.together(sp.expand(-4 * (2 * f[(2, 0)] - f[(0, 2)])))
    n5 = sp.together(sp.expand(-72 * (f[(2, 1)] - f[(0, 3)])))
    # exact assertions (the trap tests — abort import on any mismatch)
    assert sp.simplify(g2 - 1 / _M2**2) == 0, "g2 row failed kernel check"
    assert sp.simplify(g3 - (3 - 2 * _X) / _M2**3) == 0, "g3 row failed"
    assert sp.simplify(g4s - 1 / (2 * _M2**4)) == 0, "g4(s^2) row failed"
    # the two g_4 representations differ by exactly n4/32 (d=4)
    assert sp.simplify(32 * (g4t - g4s) - n4) == 0, "g4(t^2)-g4(s^2) != n4/32"
    assert sp.simplify(n4 - 2 * _X * (_X - 8) / _M2**4) == 0, "n4 failed"
    assert sp.simplify(n5 - 2 * _X * (2 * _X**2 - 43 * _X + 150) / _M2**5) == 0, \
        "n5 failed"
    # pointwise-zero (trivial s->-s-t symmetry) combos must be identically zero
    assert sp.simplify(f[(2, 1)] - f[(1, 2)]) == 0, "s2t-st2 not pointwise zero"
    assert sp.simplify(f[(3, 0)]) == 0, "s^3 coeff not pointwise zero"
    return {"g2": (sp.Integer(1), 2), "g3": (3 - 2 * _X, 3),
            "n4": (2 * _X * (_X - 8), 4),
            "n5": (2 * _X * (2 * _X**2 - 43 * _X + 150), 5)}


_ROWS = _derived_rows()          # verified at import time
_ROWS["g4"] = (sp.Rational(1, 2), 4)          # g_4 = <1/(2 m^8)>, spin-blind
_TARGET_ROWS = {"gt3": ("g2", "g3"), "gt4": ("g2", "g4")}
_NULLGEN_MAX = 0                 # highest Mandelstam order generated so far


def ensure_nulls(n_max: int):
    """Generate ALL independent crossing null constraints with Mandelstam
    order n <= n_max into _ROWS (names n4, n5, n6, n7a, n7b, ...). Machinery
    (validated: reproduces n4/n5/n6 exactly and the literature counts
    1,1,1,2,2,2,3 for n=4..10): B_k(t) rules from the residue identity
      g-side  = Res_{s=0}[(2s+t)/(s(s+t)) * M_low/[s(s+t)]^{k/2}],
      heavy   = t-expansion of (2m^2+t)/(m^2+t) P_J(1+2t/m^2)/[m^2(m^2+t)]^{k/2};
    order-n null constraints = nullspace of the g-side coefficient matrix over
    the degree-n couplings PLUS the g^2/lambda pole columns (pole leakage is
    killed too, not assumed absent). Every null is asserted HOMOGENEOUS
    (= P(X)/m^{2n}); pointwise-zero (trivial s->-s-t) combos are discarded.
    Returns the sorted name list of all nulls with order <= n_max."""
    global _NULLGEN_MAX
    if n_max > _NULLGEN_MAX:
        lam, gsq = sp.symbols("_lam _gsq")
        sig2 = _S**2 + _T**2 + (_S + _T)**2
        sig3 = -_S * _T * (_S + _T)
        coups = {(p, q): 2 * p + 3 * q
                 for p in range(0, n_max // 2 + 1)
                 for q in range(0, n_max // 3 + 1)
                 if 2 <= 2 * p + 3 * q <= n_max}
        gsyms = {pq: sp.Symbol(f"_g_{pq[0]}_{pq[1]}") for pq in coups}
        M_low = (-gsq * (1 / _S + 1 / _T - 1 / (_S + _T)) - lam
                 + sum(gsyms[pq] * sig2**pq[0] * sig3**pq[1] for pq in coups))
        rules = {}
        for k in range(2, n_max + 1, 2):
            integ = ((2 * _S + _T) / (_S * (_S + _T))
                     * M_low / (_S * (_S + _T))**(k // 2))
            ser = sp.series(integ * _S**(k // 2 + 2), _S, 0,
                            k // 2 + 3).removeO()
            res = sp.expand(ser.coeff(_S, k // 2 + 1))
            res_t = sp.series(res, _T, 0, n_max - k + 1).removeO()
            ker = ((2 * _M2 + _T) / (_M2 + _T)
                   * _pj_taylor(2 * _T / _M2, _X, n_max - k)
                   / (_M2 * (_M2 + _T))**(k // 2))
            ker_t = sp.series(sp.expand(ker), _T, 0, n_max - k + 1).removeO()
            for r in range(0, n_max - k + 1):
                rules[(k, r)] = (sp.expand(res_t.coeff(_T, r)),
                                 sp.expand(ker_t.coeff(_T, r)))
        for n in range(max(4, _NULLGEN_MAX + 1), n_max + 1):
            rl = [kr for kr in rules if kr[0] + kr[1] == n]
            cols = [gsyms[pq] for pq, deg in coups.items()
                    if deg == n] + [gsq, lam]
            A = sp.Matrix([[rules[kr][0].coeff(c) for c in cols]
                           for kr in rl])
            for i, kr in enumerate(rl):    # no cross-order leakage allowed
                left = rules[kr][0] - sum(A[i, j] * c
                                          for j, c in enumerate(cols))
                assert sp.expand(left) == 0, f"cross-order leakage at {kr}"
            indep, suffix = [], "abcdefgh"
            for v in A.T.nullspace():
                hv = sp.expand(sum(v[i] * rules[rl[i]][1]
                                   for i in range(len(rl))))
                if hv == 0:
                    continue               # trivial-symmetry combo
                px = sp.expand(hv * _M2**n)
                assert not px.has(_M2), f"null at n={n} not homogeneous"
                poly = sp.Poly(px, _X)
                cont = sp.gcd([sp.Rational(c) for c in poly.all_coeffs()
                               if c != 0])
                px = sp.expand(px / cont)
                if sp.LC(sp.Poly(px, _X)) < 0:
                    px = sp.expand(-px)
                if any(sp.simplify(px - e) == 0 for e in indep):
                    continue
                indep.append(px)
            base = sp.Matrix([[sp.Poly(e, _X).coeff_monomial(_X**d)
                               for d in range(n + 1)] for e in indep])
            assert base.rank() == len(indep), f"dependent nulls at n={n}"
            for i, e in enumerate(indep):
                name = f"n{n}" if len(indep) == 1 else f"n{n}{suffix[i]}"
                if name not in _ROWS:
                    _ROWS[name] = (e, n)
        # regression anchors (never regenerate wrong)
        assert sp.expand(_ROWS["n4"][0]
                         - 2 * _X * (_X - 8)) == 0
        assert sp.expand(_ROWS["n5"][0]
                         - 2 * _X * (2 * _X**2 - 43 * _X + 150)) == 0
        if n_max >= 6:
            _r = sp.simplify(_ROWS["n6"][0] / (2 * _X**4 - 88 * _X**3
                             + 1176 * _X**2 - 4896 * _X))
            assert _r.is_number and _r > 0, "n6 anchor failed"
        _NULLGEN_MAX = n_max
    return sorted((nm for nm in _ROWS
                   if nm.startswith("n") and _ROWS[nm][1] <= n_max),
                  key=lambda nm: (_ROWS[nm][1], nm))


def _numerator_rows(nulls: tuple, target: str = "gt3"):
    """Rows as exact sympy polys in (x, X) after m^2 = 1+x and clearing
    (1+x)^{k_max}. Order: (*target rows, *nulls). Also returns the exact Regge
    block rows q_i(xi) = lim_{x->inf} row_i(x, X=xi(1+x))/(1+x)^{k_max-2}."""
    names = list(_TARGET_ROWS[target]) + list(nulls)
    k_max = max(_ROWS[n][1] for n in names)
    rows, regge = [], []
    for n in names:
        expr, k = _ROWS[n]
        row = sp.expand(expr * (1 + _x)**(k_max - k))
        rows.append(sp.Poly(row, _x, _X))
        q = sp.limit(row.subs(_X, _XI * (1 + _x)) / (1 + _x)**(k_max - 2),
                     _x, sp.oo)
        regge.append(sp.Poly(sp.expand(q), _XI))
    return names, k_max, rows, regge


# -- spec / verifier ----------------------------------------------------------- #
@dataclass
class PositivitySpec:
    """An extremal-bound problem for the scalar-EFT dispersive SDP.

    target:  'gt3' (= g_3 M^2 / g_2), side 'lower' or 'upper'.
    nulls:   which null constraints span the functional ('n4', 'n5', ...).
             This is the FUNCTIONAL DESIGN SPACE for the structural search.
    j_max:   spin truncation inside SDPB (audit makes the result sound).
    j_audit: exact per-spin certification range for the a-posteriori audit.
    """

    target: str = "gt3"
    side: str = "lower"              # 'lower' | 'upper'
    nulls: tuple = ("n4",)
    # if set, OVERRIDE nulls with ALL independent null constraints of
    # Mandelstam order n <= null_order (the standard convergence ladder)
    null_order: Optional[int] = None
    j_max: int = 40
    j_audit: int = 200
    prec_bits: int = 768
    dps: int = 220

    def resolved_nulls(self) -> tuple:
        if self.null_order is not None:
            return tuple(ensure_nulls(self.null_order))
        return self.nulls

    def validate(self):
        if self.target not in _TARGET_ROWS:
            raise ValueError(f"unknown target {self.target}")
        if self.side not in ("lower", "upper"):
            raise ValueError("side must be lower|upper")
        if self.null_order is not None and not (4 <= self.null_order <= 20):
            raise ValueError("null_order must be in [4, 20]")
        for n in self.resolved_nulls():
            if n not in _ROWS or n in ("g2", "g3"):
                raise ValueError(f"unknown null constraint {n}")
        if len(set(self.nulls)) != len(self.nulls):
            raise ValueError("duplicate null constraints")
        if self.j_max < 4 or self.j_max % 2:
            raise ValueError("j_max must be even and >= 4")


class PositivityVerifier:
    """Certified extremal bounds on scalar-EFT Wilson-coefficient ratios."""

    def extremal_bound(self, spec: PositivitySpec) -> dict:
        spec.validate()
        _ensure_env()
        mp.mp.dps = spec.dps
        nulls = spec.resolved_nulls()
        names, k_max, rows, regge = _numerator_rows(nulls, spec.target)
        pmp = self._build_pmp(spec, rows, regge)
        raw = self._solve(pmp, spec, n_comp=len(names))
        if raw["terminateReason"] != "found primal-dual optimal solution":
            raise RuntimeError(
                f"SDPB did not reach optimality: {raw['terminateReason']} "
                "(no bound is reported — 3-way discipline)")
        z = self._reconstruct(spec, raw["y"], len(names))
        # self-consistency: recomputed objective must match SDPB's
        obj = -z[0]
        if abs(float(obj) - float(raw["dualObjective"])) > 1e-6 * max(
                1.0, abs(float(raw["dualObjective"]))):
            raise RuntimeError("functional reconstruction inconsistent with "
                               "SDPB objective — refusing to certify")
        audit = self._audit(spec, rows, z)
        bound = audit["certified_value"]
        stmt = (f"gt3 >= {bound}" if spec.side == "lower" else
                f"gt3 <= {bound}")
        return {
            "target": spec.target, "side": spec.side, "nulls": nulls,
            "bound": float(bound), "bound_exact": str(bound),
            "statement": stmt,
            "sdpb_objective": float(raw["dualObjective"]),
            "functional": [str(v) for v in z],
            "audit": audit,
            "scope": audit["scope"],
            "j_max": spec.j_max,
        }

    # -- PMP construction --------------------------------------------------- #
    def _build_pmp(self, spec: PositivitySpec, rows, regge):
        n_comp = len(rows)
        # objective b.z = -z_g2 (maximized). lower: z_g3 = +1, bound = -z_g2;
        # upper: z_g3 = -1, bound = +z_g2.
        objective = ["-1"] + ["0"] * (n_comp - 1)
        norm = ["0"] * n_comp
        norm[1] = "1" if spec.side == "lower" else "-1"
        base = mp.nstr(mp.e**-1, spec.dps, strip_zeros=False)

        def poly_strs(p: sp.Poly, var):
            cs = [sp.Rational(c) for c in reversed(sp.Poly(p, var).all_coeffs())]
            return [mp.nstr(mp.mpf(c.p) / mp.mpf(c.q), spec.dps,
                            strip_zeros=False) for c in cs]

        blocks = []
        for J in range(0, spec.j_max + 1, 2):
            Xv = J * (J + 1)
            vec = [poly_strs(sp.Poly(r.as_expr().subs(_X, Xv), _x), _x)
                   for r in rows]
            blocks.append({"DampedRational": {"constant": "1", "base": base,
                                              "poles": []},
                           "polynomials": [[vec]]})
        # exact Regge block in xi >= 0 (lossless: a limit of the constraint set)
        vec = [poly_strs(q, _XI) for q in regge]
        blocks.append({"DampedRational": {"constant": "1", "base": base,
                                          "poles": []},
                       "polynomials": [[vec]]})
        return {"objective": objective, "normalization": norm,
                "PositiveMatrixWithPrefactorArray": blocks}

    # -- solve (clone of the modular pattern, optimization mode) ------------- #
    def _solve(self, pmp: dict, spec: PositivitySpec, n_comp: int) -> dict:
        os.environ.setdefault("QGSE_SDPB_IMAGE",
                              "bootstrapcollaboration/sdpb:3.0.0")
        with _in_dir(Path("qgse_positivity_work") / str(os.getpid())):
            for stale in ("qgse_pos", "qgse_pos.ck", "qgse_pos_out"):
                if Path(stale).exists():
                    shutil.rmtree(stale)
            Path("pmp.json").write_text(json.dumps(pmp))
            nproc = os.environ.get("QGSE_SDPB_NPROC", "4")
            with open("solver.log", "w") as log:
                subprocess.check_call(
                    ["mpirun", "-n", nproc, "pmp2sdp", "-f", "json",
                     "-i", "pmp.json", "-o", "qgse_pos", "-p",
                     str(spec.prec_bits)], stdout=log,
                    stderr=subprocess.STDOUT)
                subprocess.check_call(
                    ["mpirun", "-n", nproc, "sdpb", "-s", "qgse_pos",
                     f"--precision={spec.prec_bits}"],
                    stdout=log, stderr=subprocess.STDOUT)
            out = {"terminateReason": "unknown", "dualObjective": None,
                   "y": None}
            otxt = Path("qgse_pos_out/out.txt")
            if otxt.exists():
                for line in otxt.read_text().splitlines():
                    if "=" not in line:
                        continue
                    key, val = [p.strip() for p in line.split("=", 1)]
                    val = val.rstrip(";").strip().strip('"')
                    if key == "terminateReason":
                        out["terminateReason"] = val
                    elif key == "dualObjective":
                        out["dualObjective"] = mp.mpf(val)
            ytxt = Path("qgse_pos_out/y.txt")
            if ytxt.exists():
                vals = ytxt.read_text().split()
                nrows = int(vals[0])         # header: "rows cols" then values
                out["y"] = [mp.mpf(v) for v in vals[2:2 + nrows]]
            if out["y"] is not None and len(out["y"]) != n_comp - 1:
                raise RuntimeError(
                    f"y.txt has {len(out['y'])} entries, expected "
                    f"{n_comp - 1} (normalization eliminates one)")
        return out

    @staticmethod
    def _reconstruct(spec: PositivitySpec, y: list, n_comp: int) -> list:
        """Rebuild the full functional z from SDPB's reduced y: the
        normalization has a single nonzero on component 1 (z_g3 = +-1),
        which pmp2sdp eliminates; y = (z_0, z_2, z_3, ...) in order.
        Verified by the objective self-consistency check in extremal_bound."""
        fixed = mp.mpf(1) if spec.side == "lower" else mp.mpf(-1)
        return [y[0], fixed] + list(y[1:])

    # -- exact a-posteriori audit (what makes the bound sound) --------------- #
    def _audit(self, spec: PositivitySpec, rows, z) -> dict:
        """Round the functional to rationals and re-verify positivity in
        EXACT arithmetic. Rounding can nick the extremal double roots, so a
        SOUND margin backoff is applied: z_g2 += eps adds eps*(1+x)^{k-2} > 0
        (strictly more positive) while WEAKENING the reported bound — the
        safe direction on both sides. First eps whose functional passes wins;
        none passing = hard error (never certify a failing functional)."""
        zr0 = [_mpf_to_rational(v) for v in z]
        scale = max(sp.Rational(1), abs(zr0[0]))
        last_fail = "no eps attempted"
        for eps_exp in (0, 9, 7, 5, 3):
            eps = sp.Rational(0) if eps_exp == 0 else scale / 10**eps_exp
            zr = list(zr0)
            zr[0] = zr0[0] + eps
            ok, tail_proven, degX, why = self._audit_once(spec, rows, zr)
            if ok:
                break
            last_fail = why
        else:
            raise RuntimeError(
                f"audit FAILED at every rounding margin: {last_fail} — "
                "spurious bound refused (J-truncation / rounding trap)")
        if tail_proven:
            scope = "all spins (complete exact certificate)"
        else:
            scope = (f"spectra with all spins J <= {spec.j_audit} "
                     f"(exact per-spin certificate; deg_X={degX} tail proof "
                     "not implemented — scoped, not assumed)")
        certified = -zr[0] if spec.side == "lower" else zr[0]
        return {"certified_value": certified, "rational_functional":
                [str(v) for v in zr], "per_spin_checked_to": spec.j_audit,
                "margin_eps": str(zr[0] - zr0[0]),
                "tail_proven": tail_proven, "scope": scope}

    @staticmethod
    def _audit_once(spec: PositivitySpec, rows, zr):
        """One exact positivity pass for a rational functional zr.
        Returns (ok, tail_proven, degX, why_failed)."""
        G = sum(zr[i] * rows[i].as_expr() for i in range(len(rows)))
        GX = sp.Poly(sp.expand(G), _X)
        for J in range(0, spec.j_audit + 1, 2):
            gj = sp.Poly(sp.expand(G.subs(_X, J * (J + 1))), _x)
            if not _poly_nonneg_on_halfline(gj):
                return False, False, GX.degree(), f"negative at spin J={J}"
        degX = GX.degree()
        tail_proven = False
        if degX == 2:
            c2, c1, c0 = [sp.Poly(c, _x) for c in GX.all_coeffs()]
            disc = sp.Poly(sp.expand(c1.as_expr()**2
                                     - 4 * c2.as_expr() * c0.as_expr()), _x)
            x0 = _halfline_negativity_onset(disc)
            if (x0 is not None and sp.LC(c2) > 0 and c2.eval(0) > 0
                    and _poly_nonneg_on_halfline(c2)):
                cauchy = 1 + _sup_ratio_on_interval(
                    [c1, c0], c2, sp.Interval(sp.Rational(0), x0))
                j_need = 0
                while j_need * (j_need + 1) < cauchy:
                    j_need += 2
                if j_need <= spec.j_audit:
                    tail_proven = True
        return True, tail_proven, degX, ""


# -- exact polynomial positivity helpers (QQ, Sturm/sqf-based) ----------------- #
def _mpf_to_rational(v) -> sp.Rational:
    """EXACT mpf -> Rational via the binary mantissa/exponent — no tolerance.
    (nsimplify-with-tolerance rounding was refuted in practice: high-order
    functionals have 20+ near-degenerate components whose cancellations
    amplify 1e-12 rounding into genuine negativity, making the audit refuse
    sound bounds. Auditing the solver's exact value costs nothing but bigger
    rationals.)"""
    sign, man, exp, _ = mp.mpf(v)._mpf_
    r = sp.Rational(int(man)) * sp.Rational(2)**int(exp)
    return -r if sign else r
def _cauchy_bound(p: sp.Poly) -> sp.Rational:
    """Exact rational R: all real roots of p lie in [-R, R]."""
    cs = p.all_coeffs()
    lc = sp.Rational(cs[0])
    return 1 + max((abs(sp.Rational(c)) / abs(lc) for c in cs[1:]),
                   default=sp.Rational(0))


def _poly_nonneg_on_halfline(p: sp.Poly) -> bool:
    """Exact: p(x) >= 0 for all x >= 0 (p over QQ). Sign of LC at +inf plus
    no sign change in (0, inf): a change requires a root of ODD multiplicity,
    detected exactly via square-free decomposition + Sturm root counting."""
    if p.is_zero:
        return True
    if p.eval(0) < 0 or sp.LC(p) < 0:
        return False
    _, sqf = p.sqf_list()
    for q, e in sqf:
        if e % 2 == 0:
            continue
        R = _cauchy_bound(q)
        n = q.count_roots(0, R)
        if q.eval(0) == 0:
            n -= 1
        if n > 0:
            return False
    return True


def _halfline_negativity_onset(disc: sp.Poly):
    """Exact rational x0 >= 0 with disc(x) < 0 for ALL x > x0, or None.
    Uses the Cauchy bound: beyond it disc has the sign of its leading
    coefficient (loose but exact — the strip Cauchy spin-bound absorbs it)."""
    if disc.is_zero or sp.LC(disc) >= 0:
        return None
    x0 = _cauchy_bound(disc)
    return x0 if disc.eval(x0) < 0 else None


def _sup_ratio_on_interval(nums, den, iv) -> sp.Rational:
    """Rational upper bound for max_i sup_{x in iv} |num_i(x)| / den(x),
    den constant or positive poly; coarse (interval endpoint + coefficient
    bound) but exact — used only for the Cauchy spin cutoff."""
    hi = sp.Rational(iv.end)
    best = sp.Rational(0)
    dmin = min(den.eval(sp.Rational(iv.start)), den.eval(hi))
    if den.degree() > 0:
        crit = [r for r in den.real_roots() if iv.start <= r <= hi]
        if crit:
            dmin = min(dmin, *[den.eval(sp.Rational(sp.nsimplify(
                r, rational=True, tolerance=1e-20))) for r in crit])
    if dmin <= 0:
        raise RuntimeError("audit: denominator not positive on strip")
    for p in nums:
        cb = sum(abs(sp.Rational(c)) * hi**k
                 for k, c in enumerate(reversed(p.all_coeffs())))
        best = max(best, cb / dmin)
    return best
