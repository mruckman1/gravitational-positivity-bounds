"""D=10 maximal-supergravity R^4 bound — the EXTREMALITY instrument.

Target (CMRS Sec 3.8, eqs 71-79): for the susy-reduced crossing-symmetric
amplitude M = 8piG/(stu) + g_0 + ..., with the STRONGER Regge assumption
s^2 M -> 0 (flagged in every certificate — this is an additional axiom):
    0 <= g_0 <= c * 8piG/M^6,   CMRS: c = 3.000 (14 functionals).
The type II string sits INSIDE at g_0 M^6/(8piG) = 2 zeta(3) ~ 2.4041.

WHY THIS INSTRUMENT MATTERS: every certified tightening of c toward 2.4041
is quantitative evidence that consistency alone pins quantum gravity to the
string answer (the uniqueness/extremality question); a floor above 2.4041
is evidence for room beyond string theory. This module reproduces the
published c ~ 3.000 with our exact-audit standard; Phase 3 puts the LLM on
tightening it.

Sum rules (validated: C_0^imp corner-regular at u=-m^2 for even J via
P_J(-1)=1; C_-2 kernel polynomial):
  C_-2  (eq 73): 8piG/(-u) = < m^2 (2m^2+u) P_J(1+2u/m^2) >
  C_0^imp (eq 77): g_0    = < (2m^2+u) P_J(1+2u/m^2)/(m^2+u)
                              - 2u^2/(m^4-u^2) >
  X_2 (eq 51, D=10): exact nulls (zero low side).
Functional: F = int f(p) C_-2 + int h(p) C_0^imp + int e(p) X_2 at u=-p^2;
state-wise F >= 0 implies gamma_G + gamma_0 g_0 >= 0 with EXACT rational
columns gamma_G = sum a_n/(n-1), gamma_0 = sum h_i/(i+1).
Upper bound: gamma_0 = -1, minimize gamma_G. Lower: gamma_0 = +1.

Pipeline identical to gravity_lp (LP shapes, exact audit certifies,
J-scoped statements); units 8piG = 1, M = 1.
"""

from __future__ import annotations

import functools
import time

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import linprog

from qgse.verifiers.gravity_positivity import pj_poly, _M2, _U, _P
from qgse.verifiers.gravity_lp import (GravityRayVerifier, _xk_expr,
                                       _bessel_transform_fn, _W,
                                       _kernel_expr)

D = 10
F_POWERS = (2, 3, 4, 5, 6, 7, 8, 9, 10)    # f on C_-2 (needs n>1 for 1/p^2)
H_POWERS = (0, 1, 2, 3)                    # h on C_0^imp
E_POWERS = (0, 1)                          # e on X_2


# Kernel builders below are PURE in J (D is a module constant): deterministic
# sympy exprs, immutable, read-only downstream => memoizing is byte-identical
# to recomputing. Without the cache each is rebuilt once per smear-power in
# _columns, and again in both rows_grid and audit, every refine iteration.
@functools.lru_cache(maxsize=None)
def cm2_expr(J: int):
    """C_-2 kernel at u=-p^2 (polynomial)."""
    return sp.expand((_M2 * (2 * _M2 + _U)
                      * pj_poly(J, D, 1 + 2 * _U / _M2)).subs(_U, -_P**2))


@functools.lru_cache(maxsize=None)
def c0imp_expr(J: int):
    """C_0^imp kernel at u=-p^2 (corner-regular, (m2+p^2) class)."""
    pj = pj_poly(J, D, 1 + 2 * _U / _M2)
    k = (2 * _M2 + _U) * pj / (_M2 + _U) - 2 * _U**2 / (_M2**2 - _U**2)
    return sp.cancel(sp.together(k.subs(_U, -_P**2)))


@functools.lru_cache(maxsize=None)
def c2imp_susy_expr(J: int):
    """C_2^imp kernel at D=10 (heavy side identical to the D=7-family
    _kernel_expr; only the low-side bookkeeping differs: 2 g_2/(i+1)
    + g_3/(i+3) per p^i smear, NO gravity 1/p^2 term)."""
    return _kernel_expr(J, D)


@functools.lru_cache(maxsize=None)
def e1_expr(J: int):
    """E_1 null kernel (first derivative of the master crossing identity,
    CMRS eq 75 family): exact null for susy-sector heavy densities."""
    pj = pj_poly(J, D, 1 + 2 * _U / _M2)
    PJp1 = sp.Rational(J * (J + 7), 8)
    E1 = (_U * (2 * _M2 + _U) * pj / (_M2 * (_M2 + _U)**2)
          - _M2 / (_M2 - _U) * ((1 + 4 * PJp1) / (_M2 + _U)
                                - 2 * _M2 / (_M2 + _U)**2))
    return sp.cancel(sp.together(E1.subs(_U, -_P**2)))


@functools.lru_cache(maxsize=None)
def e2_expr(J: int):
    """E_2 null kernel (second s-derivative at s=0 of the master crossing
    identity <K(s,u)-K(u,s)> = 0 with K = m^2(2m^2+u)P_J(1+2u/m^2)
    /((m^2-s)(m^2+s+u)), CMRS eq 75 family; E_0 == c0imp-2, E_1 == e1_expr —
    both matches verified exactly, scripts/verify_e2_family.py): exact null
    for susy-sector heavy densities under the same s^2 M -> 0 Regge axiom.
    Same denominator class as E_1 (all (m^2-p^2) factors cancel)."""
    pj = pj_poly(J, D, 1 + 2 * _U / _M2)
    PJp1 = sp.Rational(J * (J + 7), 8)                          # P'_J(1)
    PJpp1 = sp.Rational(J * (J + 7) * (J * (J + 7) - 8), 80)    # P''_J(1)
    E2 = (2 * _M2 * pj * (1 / _M2**3 + 1 / (_M2 + _U)**3)
          - _M2 / (_M2 - _U) * (4 * (PJp1 + 2 * PJpp1) / (_M2 * (_M2 + _U))
                                - 2 * (1 + 4 * PJp1) / (_M2 + _U)**2
                                + 4 * _M2 / (_M2 + _U)**3))
    return sp.cancel(sp.together(E2.subs(_U, -_P**2)))


class SusyR4Verifier(GravityRayVerifier):
    """g_0 bound for D=10 susy-reduced gravity. Inherits the exact audit
    machinery; the COLUMN STRUCTURE and solver policy are the Phase-3
    design space (LLM-searchable)."""

    def __init__(self, f_powers=F_POWERS, h_powers=H_POWERS,
                 e_powers=E_POWERS, x4_powers=(), x6_powers=(),
                 k_powers=(), e1_powers=(), p_max=None,
                 x8_powers=(), x10_powers=(), e2_powers=()):
        self.f_powers = tuple(f_powers)
        self.h_powers = tuple(h_powers)
        self.e_powers = tuple(e_powers)
        self.x4_powers = tuple(x4_powers)
        self.x6_powers = tuple(x6_powers)
        # higher crossing nulls X_8, X_10 (same _xk_kernel machinery, k=8,10;
        # validated J=0-null + regular at u=-m^2 in the assertion battery).
        # Extra FREE (zero-low-side) constraint-satisfier directions — the
        # margin-carrying structure the tangency-floor finding calls for.
        self.x8_powers = tuple(x8_powers)
        self.x10_powers = tuple(x10_powers)
        self.k_powers = tuple(k_powers)      # C_2^imp-susy (g_2/g_3 low side)
        self.e1_powers = tuple(e1_powers)    # E_1 nulls (zero low side)
        # E_2 nulls (second derivative of the master identity; zero low side;
        # appended at the absolute END of the column order so no existing
        # offsets move): [f, h, e, x4, x6, k, e1, x8, x10, e2].
        self.e2_powers = tuple(e2_powers)
        # EXTENDED-u DOMAIN (CMRS App. B): smear p in (0, p_max], u = -p^2
        # down to -p_max^2. p_max > 1 requires the MEROMORPHY+DISCRETENESS
        # axiom (flagged in certificates). Must be exact Rational for the
        # audit. Kernels UNCHANGED (u=-m^2 pole removable, verified).
        self.p_max = sp.Rational(1) if p_max is None else sp.Rational(p_max)
        assert self.p_max >= 1
        self.viability_rows = False   # Step-1 large-J/b-extension shaping:
        # opt-in (post-campaign upgrade pass); default = the certified-
        # reliable pipeline that produced c = 3.0136
        self._col_cache = {}          # J -> column list (see _columns)

    def _columns(self, J):
        # Columns depend ONLY on J and the *_powers tuples + p_max, all set
        # in __init__ and never mutated (viability_rows toggles rows_grid's
        # b-grid, not the columns). The returned sympy exprs are immutable
        # and every caller reads them only (lambdify / weighted sum), so a
        # per-instance J-keyed memo is byte-identical to recomputing — and
        # removes the rows_grid-vs-audit double build each refine iteration.
        cached = self._col_cache.get(J)
        if cached is not None:
            return cached
        # NOTE: x8/x10 (nulls) are appended AT THE END so the k_powers /
        # e1_powers column offsets used by solve()/certify_g0_upper() (the
        # exact g_2=g_3=0 projection) stay correct. Order:
        #   [f, h, e, x4, x6, k, e1, x8, x10].
        cols = ([_P**n * cm2_expr(J) for n in self.f_powers]
                + [_P**i * c0imp_expr(J) for i in self.h_powers]
                + [_P**i * _xk_expr(2, J, D) for i in self.e_powers]
                + [_P**i * _xk_expr(4, J, D) for i in self.x4_powers]
                + [_P**i * _xk_expr(6, J, D) for i in self.x6_powers]
                + [_P**i * c2imp_susy_expr(J) for i in self.k_powers]
                + [_P**i * e1_expr(J) for i in self.e1_powers]
                + [_P**i * _xk_expr(8, J, D) for i in self.x8_powers]
                + [_P**i * _xk_expr(10, J, D) for i in self.x10_powers]
                + [_P**i * e2_expr(J) for i in self.e2_powers])
        self._col_cache[J] = cols
        return cols

    def rows_grid(self, j_max, n_xgrid, b_grid, extra_pts=()):
        tags, rows = [], []
        xs = np.linspace(0.0, 1.0 - 1.0 / n_xgrid, n_xgrid)
        m2s = 1.0 / (1.0 - xs)
        pm = float(self.p_max)
        pgrid = np.linspace(1e-4, pm, max(240, int(240 * pm)))
        for J in range(0, j_max + 1, 2):
            fns = [sp.lambdify((_M2, _P), c, "numpy")
                   for c in self._columns(J)]
            for m2 in list(m2s) + [m for (Jx, m) in extra_pts if Jx == J]:
                # finite-guard: higher-k nulls (X_8/X_10) have higher-order
                # removable singularities at the p~m corner that lambdify can
                # render as inf/nan; zero them before the trapezoid (SHAPING
                # rows only — the audit is exact, so this cannot affect a
                # certificate). Identity for the existing kernels (no inf/nan
                # on the grid), so champion certificates are unchanged.
                row = [np.trapezoid(np.nan_to_num(np.broadcast_to(
                          np.asarray(fn(m2, pgrid), dtype=float),
                          pgrid.shape), nan=0.0, posinf=0.0, neginf=0.0),
                          pgrid) for fn in fns]
                rows.append(row)
                tags.append((J, float(m2)))
        # m->inf closure: C_-2 dominates (grows m^4) -> Bessel rows on f only.
        # ENDPOINT: the closure integrates f over the FULL smearing range
        # (0, p_max]; the default pmax=1.0 silently truncated extended-u rows
        # (shaping-only, so no certificate was affected — found 2026-07-16 in
        # the E1' build; true-endpoint f_hat re-measured clean for the ledgered
        # margin functional).
        lo, hi, cnt = b_grid
        tf = [_bessel_transform_fn(n, D, float(self.p_max))
              for n in self.f_powers]
        nz = (len(self.h_powers) + len(self.e_powers)
              + len(self.x4_powers) + len(self.x6_powers)
              + len(self.x8_powers) + len(self.x10_powers)
              + len(self.k_powers) + len(self.e1_powers)
              + len(self.e2_powers))
        # (1a) dense grid to B=40 + extension [40, 160] at delta_b = 0.5
        bs = (np.concatenate([np.linspace(lo, 40.0, int(cnt)),
                              np.arange(40.5, 160.01, 0.5)])
              if self.viability_rows else np.linspace(lo, hi, int(cnt)))
        for b in bs:
            rows.append([t(float(b)) for t in tf] + [0.0] * nz)
            tags.append(("bessel", float(b)))
        # (1b) asymptotic sign row: A(b) -> 16 a_2 / b^3 must be positive
        # (kills the champion's b > 160 persistent-negative failure)
        if self.viability_rows:
            a2row = [16.0 if n == 2 else 0.0
                     for n in self.f_powers] + [0.0] * nz
            rows.append(a2row)
            tags.append(("bessel", "asym_a2"))
        return np.array(rows), tags

    # -- stable large-J machinery (ported from the recon's verified
    #    ejtail.py: jet-subtracted factored kernels; expanded pj_poly
    #    lambdify is numerical garbage for J >~ 60) ------------------------
    @staticmethod
    def _pr1(r, J):
        import mpmath as _mp
        out = _mp.mpf(1)
        for i in range(r):
            out *= (J * (J + 7) - i * (i + 7)) / _mp.mpf(2 * (4 + i))
        return out

    @staticmethod
    def _rho(k, r, m2, u):
        if (k, r) == (2, 0): return (2*m2+u)/(m2*u**2*(m2+u)**2)
        if (k, r) == (2, 1): return -4/(m2*u*(u-m2)*(m2+u))
        if (k, r) == (4, 0): return (2*m2+u)/(m2**2*u**3*(m2+u)**3)
        if (k, r) == (4, 1): return 2*(-2*m2**2+3*m2*u+3*u**2)/(m2**4*u**2*(u-m2)*(m2+u)**2)
        if (k, r) == (4, 2): return -4/(m2**4*u*(u-m2)*(m2+u))
        if (k, r) == (6, 0): return (2*m2+u)/(m2**3*u**4*(m2+u)**4)
        if (k, r) == (6, 1): return -2*(2*m2**4-3*m2**3*u+2*m2**2*u**2+10*m2*u**3+5*u**4)/(m2**7*u**3*(u-m2)*(m2+u)**3)
        if (k, r) == (6, 2): return 2*(-2*m2**2+5*m2*u+5*u**2)/(m2**7*u**2*(u-m2)*(m2+u)**2)
        if (k, r) == (6, 3): return -8/(3*m2**7*u*(u-m2)*(m2+u))
        # k=8 entries: derived by the same Laurent-of-W_k construction that
        # reproduces the k=2,4,6 tables exactly; verified to 5.7e-51 against
        # exact _xk_expr(8,J,10) (scripts/verify_x8_stable.py, all PASS).
        if (k, r) == (8, 0): return (2*m2+u)/(m2**4*u**5*(m2+u)**5)
        if (k, r) == (8, 1): return -2*(2*m2**6-3*m2**5*u+2*m2**4*u**2-2*m2**3*u**3-31*m2**2*u**4-36*m2*u**5-12*u**6)/(m2**10*u**4*(u-m2)*(m2+u)**4)
        if (k, r) == (8, 2): return -2*(2*m2**4-5*m2**3*u+7*m2**2*u**2+24*m2*u**3+12*u**4)/(m2**10*u**3*(u-m2)*(m2+u)**3)
        if (k, r) == (8, 3): return 4*(-2*m2**2+7*m2*u+7*u**2)/(3*m2**10*u**2*(u-m2)*(m2+u)**2)
        if (k, r) == (8, 4): return -4/(3*m2**10*u*(u-m2)*(m2+u))
        raise KeyError((k, r))

    @staticmethod
    def _G(k, r, m2, p2):
        """Factored regular kernels G_{k,r} = rho0 (x-1)^r/r! - rho_r
        (recon-verified; no forward-limit cancellation)."""
        if (k, r) == (2, 1):
            return 2*(p2 - 3*m2)/(m2**2*(m2 - p2)**2*(m2 + p2))
        if (k, r) == (4, 1):
            return 2*(5*m2 - 3*p2)/(m2**4*(m2 - p2)**3*(m2 + p2))
        if (k, r) == (4, 2):
            return -SusyR4Verifier._G(4, 1, m2, p2)
        if (k, r) == (6, 1):
            return -2*(12*m2**2 - 15*m2*p2 + 5*p2**2)/(m2**7*(m2 - p2)**4*(m2 + p2))
        if (k, r) == (6, 2):
            return -SusyR4Verifier._G(6, 1, m2, p2)
        if (k, r) == (6, 3):
            return -4*(7*m2**2 - 7*m2*p2 + 2*p2**2)/(3*m2**7*(m2 - p2)**4*(m2 + p2))
        if (k, r) == (8, 1):
            return 2*(33*m2**3 - 67*m2**2*p2 + 48*m2*p2**2
                      - 12*p2**3)/(m2**10*(m2 - p2)**5*(m2 + p2))
        if (k, r) == (8, 2):
            return -SusyR4Verifier._G(8, 1, m2, p2)
        if (k, r) == (8, 3):
            return 4*(21*m2**3 - 40*m2**2*p2 + 28*m2*p2**2
                      - 7*p2**3)/(3*m2**10*(m2 - p2)**5*(m2 + p2))
        if (k, r) == (8, 4):
            return -2*(9*m2**3 - 13*m2**2*p2 + 8*m2*p2**2
                       - 2*p2**3)/(3*m2**10*(m2 - p2)**5*(m2 + p2))
        raise KeyError((k, r))

    def _xk_stable(self, k, J, m2, u, Pj):
        import mpmath as _mp
        R = k // 2
        s_ = -2 * u / m2
        if J * (J + 7) * s_ < 20:      # jet regime: cancellation-free
            tot = _mp.mpf(0)
            for r in range(1, R + 1):
                tot += self._pr1(r, J) * self._G(k, r, m2, -u)
            r0 = self._rho(k, 0, m2, u)
            xm1 = 2 * u / m2
            for r in range(R + 1, 81):
                t = self._pr1(r, J) * r0 * xm1**r / _mp.factorial(r)
                tot += t
                if abs(t) < _mp.mpf(10)**(-50) * (1 + abs(tot)):
                    break
            return tot
        r0 = self._rho(k, 0, m2, u)
        res = sum(self._pr1(r, J) * self._rho(k, r, m2, u)
                  for r in range(R + 1))
        return r0 * Pj - res

    def _stable_row(self, J, m2, pg, pjv):
        """Full column row at (m2, J) with stable P_J values. Near-corner
        p ~ m points are masked (removable singularities of the factored
        forms; trapz interpolates across — SHAPING rows only, the audit is
        exact)."""
        import mpmath as _mp
        p2 = pg * pg
        u = -p2
        corner = np.abs(m2 - p2) < 1e-3 * m2
        cm2 = m2 * (2 * m2 - p2) * pjv
        c0 = np.where(corner, 0.0,
                      (2 * m2 - p2) * pjv / np.where(corner, 1.0, m2 - p2)
                      - 2 * p2 * p2 / np.where(corner, 1.0,
                                               m2 * m2 - p2 * p2))
        row = [np.trapezoid(pg**n * cm2, pg) for n in self.f_powers]
        row += [np.trapezoid(pg**i * c0, pg) for i in self.h_powers]
        PJp1 = J * (J + 7) / 8.0
        for fam, pws in (("x2", self.e_powers), ("x4", self.x4_powers),
                         ("x6", self.x6_powers)):
            k = {"x2": 2, "x4": 4, "x6": 6}[fam]
            vals = np.array([0.0 if c else float(self._xk_stable(
                k, J, _mp.mpf(m2), _mp.mpf(-pp*pp),
                _mp.mpf(float(pj_))))
                for pp, pj_, c in zip(pg, pjv, corner)])
            row += [np.trapezoid(pg**i * vals, pg) for i in pws]
        if self.k_powers:
            c2i = np.where(corner, 0.0,
                           (2*m2 - p2) * pjv / (m2 * np.where(corner, 1.0,
                                                              (m2 - p2))**2)
                           - (p2*p2/m2**3) * ((4*m2 - 3*p2)
                              / np.where(corner, 1.0, (m2 - p2))**2
                              - 4*p2*PJp1 / np.where(corner, 1.0,
                                                     m2*m2 - p2*p2)))
            row += [np.trapezoid(pg**i * c2i, pg) for i in self.k_powers]
        if self.e1_powers:
            e1v = np.where(corner, 0.0,
                           -p2 * (2*m2 - p2) * pjv
                           / (m2 * np.where(corner, 1.0, (m2 - p2))**2)
                           - m2/(m2 + p2) * ((1 + 4*PJp1)
                              / np.where(corner, 1.0, m2 - p2)
                              - 2*m2 / np.where(corner, 1.0,
                                                (m2 - p2))**2))
            row += [np.trapezoid(pg**i * e1v, pg) for i in self.e1_powers]
        if self.x8_powers:
            # X_8 via the k=8 stable forms (columns order: AFTER k/e1, matching
            # _columns; verified scripts/verify_x8_stable.py)
            vals = np.array([0.0 if c else float(self._xk_stable(
                8, J, mp.mpf(m2), mp.mpf(-pp*pp), mp.mpf(float(pj_))))
                for pp, pj_, c in zip(pg, pjv, corner)])
            row += [np.trapezoid(pg**i * vals, pg) for i in self.x8_powers]
        if self.e2_powers:
            # E_2 (absolute end of column order, matching _columns)
            PJpp1 = J * (J + 7) * (J * (J + 7) - 8) / 80.0
            safe1 = np.where(corner, 1.0, m2 - p2)
            e2v = np.where(corner, 0.0,
                           2 * m2 * pjv * (1.0 / m2**3 + 1.0 / safe1**3)
                           - m2 / (m2 + p2) * (4 * (PJp1 + 2 * PJpp1)
                                               / (m2 * safe1)
                                               - 2 * (1 + 4 * PJp1) / safe1**2
                                               + 4 * m2 / safe1**3))
            row += [np.trapezoid(pg**i * e2v, pg) for i in self.e2_powers]
        return row

    def solve(self, side, j_max=40, n_xgrid=300, b_grid=(0.25, 80.0, 240),
              extra_pts=(), cap=3000.0, give=0.10):
        rows, tags = self.rows_grid(j_max, n_xgrid, b_grid, extra_pts)
        nf, nh = len(self.f_powers), len(self.h_powers)
        ne = (len(self.e_powers) + len(self.x4_powers)
              + len(self.x6_powers) + len(self.k_powers)
              + len(self.e1_powers)
              + len(self.x8_powers) + len(self.x10_powers)
              + len(self.e2_powers))
        n = nf + nh + ne
        Pf = float(self.p_max)
        gG = np.array([Pf**(p - 1) / (p - 1) for p in self.f_powers]
                      + [0.0] * (nh + ne))                  # gamma_G
        g0 = np.array([0.0] * nf
                      + [Pf**(i + 1) / (i + 1) for i in self.h_powers]
                      + [0.0] * ne)                         # gamma_0
        sgn = -1.0 if side == "upper" else 1.0
        A_eq = g0.reshape(1, -1)
        b_eq = np.array([sgn])
        if self.k_powers:
            nf_h = nf + nh
            off = nf_h + len(self.e_powers) + len(self.x4_powers) \
                + len(self.x6_powers)
            Pf_ = float(self.p_max)
            g2r = np.zeros(n); g3r = np.zeros(n)
            for j, i in enumerate(self.k_powers):
                g2r[off + j] = 2.0 * Pf_**(i + 1) / (i + 1)
                g3r[off + j] = Pf_**(i + 3) / (i + 3)
            # pure-null mode: eliminate BOTH g_2 and g_3 content exactly
            A_eq = np.vstack([A_eq, g2r, g3r])
            b_eq = np.append(b_eq, [0.0, 0.0])
        res = linprog(gG, A_ub=-rows, b_ub=np.zeros(len(rows)),
                      A_eq=A_eq, b_eq=b_eq, bounds=[(-cap, cap)] * n,
                      method="highs")
        if not res.success:
            raise RuntimeError(f"LP infeasible/failed: {res.message}")
        opt = float(res.fun)
        # robust slack stage on (m,J) rows only
        scale = np.abs(rows).sum(axis=1) + 1e-12
        nrows = rows / scale[:, None]
        sl = np.array([[0.0] if t[0] == "bessel" else [1.0] for t in tags])
        c2 = np.zeros(n + 1); c2[-1] = -1.0
        A_ub2 = np.vstack([np.hstack([-nrows, sl]), np.append(gG, 0.0)])
        b_ub2 = np.append(np.zeros(len(nrows)), opt + give * abs(opt) + 1e-9)
        A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
        for cap2 in (300.0, 1000.0, cap):
            r2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq,
                         bounds=[(-cap2, cap2)] * n + [(0, None)],
                         method="highs")
            if r2.success and r2.x[-1] > 1e-9:
                return r2.x[:n], float(gG @ r2.x[:n]), float(r2.x[-1])
        return res.x, opt, 0.0

    def audit(self, a_rat, J, j_audit_spec):
        """Exact per-spin audit of the total integrand (reuses parent)."""
        p = _P
        cols = self._columns(J)
        integrand = sum(a_rat[i] * cols[i] for i in range(len(cols)))

        class _S:                                  # minimal spec shim
            powers = F_POWERS
            x_smears = ()
            j_audit = j_audit_spec
            D_ = D
        # inline the parent's audit body via _audit_spin machinery:
        Eq, r0, r1, const = self._exact_smear_integral(
            integrand, D, P=self.p_max)
        w = _W
        sub = {_M2: 1 / w**2}
        R0 = sp.cancel(sp.together(Eq.subs(sub) / const.subs(sub))) \
            if Eq != 0 else sp.Integer(0)
        R1 = sp.cancel(sp.together((r0 * w).subs(sub) / const.subs(sub))) \
            if r0 != 0 else sp.Integer(0)
        R2 = sp.cancel(sp.together(r1.subs(sub) / (2 * const.subs(sub)))) \
            if r1 != 0 else sp.Integer(0)
        return self._audit_wpieces(R0, R1, R2, J, P=self.p_max)


def certify_g0_upper(j_max=40, j_audit=40, max_refine=6, log=print,
                     verifier=None, cap=3000.0, give=0.10,
                     save_functional=None):
    """Certify g_0 <= c * 8piG/M^6 (D=10, susy Regge axiom, J-scoped).

    save_functional: optional path. On a successful certificate, persist the
    exact-rational functional a_rat (the certificate itself, which the harness
    otherwise recomputes on demand and never stores) together with the column
    config, so the certified functional can be re-inspected (e.g. per-spin tail
    behaviour) without re-running the shaping LP."""
    V = verifier if verifier is not None else SusyR4Verifier()
    extra = []
    for it in range(max_refine):
        t0 = time.time()
        a, c_val, slack = V.solve("upper", j_max=j_max, extra_pts=extra, cap=cap, give=give)
        log(f"[iter {it}] LP c = {c_val:.4f} slack={slack:.2e} "
            f"({time.time()-t0:.0f}s)")
        a_rat = V._rationalize(a)
        nf_ = len(V.f_powers)
        Pq = V.p_max
        if V.k_powers and len(V.k_powers) >= 2:
            # exact projection: solve the last two k-coefficients so that
            # gamma_2 = gamma_3 = 0 EXACTLY for the rationalized functional
            off = (nf_ + len(V.h_powers) + len(V.e_powers)
                   + len(V.x4_powers) + len(V.x6_powers))
            ks = list(V.k_powers)
            g2c = [2 * Pq**(i + 1) / (i + 1) for i in ks]
            g3c = [Pq**(i + 3) / (i + 3) for i in ks]
            m11, m12 = g2c[-2], g2c[-1]
            m21, m22 = g3c[-2], g3c[-1]
            r1 = -sum(g2c[j] * a_rat[off + j] for j in range(len(ks) - 2))
            r2 = -sum(g3c[j] * a_rat[off + j] for j in range(len(ks) - 2))
            det = m11 * m22 - m12 * m21
            a_rat[off + len(ks) - 2] = (r1 * m22 - r2 * m12) / det
            a_rat[off + len(ks) - 1] = (m11 * r2 - m21 * r1) / det
        g0c = sum(q * Pq**(i + 1) / (i + 1) for q, i in
                  zip(a_rat[nf_:nf_ + len(V.h_powers)], V.h_powers))
        a_rat = [-q / g0c for q in a_rat]          # exact gamma_0 = -1
        cG = sum(q * Pq**(n - 1) / (n - 1)
                 for q, n in zip(a_rat[:nf_], V.f_powers))
        fails = []
        t0 = time.time()
        for J in range(0, j_audit + 1, 2):
            ok, why = V.audit(a_rat, J, j_audit)
            if not ok:
                fails.append((J, why))
                log(f"    audit FAIL {why}")
                if len(fails) >= 3:
                    break
        log(f"[iter {it}] audit {'PASS' if not fails else 'FAIL'} "
            f"({time.time()-t0:.0f}s)")
        if not fails:
            if save_functional:
                import json as _json
                _json.dump(
                    {"a_rat": [str(x) for x in a_rat],
                     "c_cert_exact": str(cG),
                     "config": {"f_powers": list(V.f_powers),
                                "h_powers": list(V.h_powers),
                                "e_powers": list(V.e_powers),
                                "x4_powers": list(V.x4_powers),
                                "x6_powers": list(V.x6_powers),
                                "k_powers": list(V.k_powers),
                                "e1_powers": list(V.e1_powers),
                                "x8_powers": list(V.x8_powers),
                                "x10_powers": list(V.x10_powers),
                                "e2_powers": list(V.e2_powers),
                                "p_max": str(V.p_max)},
                     "j_audit": j_audit},
                    open(save_functional, "w"), indent=1)
                log(f"[iter {it}] SAVED functional -> {save_functional}")
            return {"c_cert": float(cG), "c_cert_exact": str(cG),
                    "statement": f"g_0 <= ({cG}) * 8piG/M^6",
                    "scope": ((f"EXTENDED-u DOMAIN p_max={V.p_max} (AXIOM: meromorphy+discreteness, CMRS App. B); " if V.p_max > 1 else "") + f"D=10 susy-reduced gravity, spins J <= "
                              f"{j_audit}; ADDITIONAL AXIOM: s^2 M -> 0 "
                              "Regge (CMRS Sec 3.8)"),
                    "string_value": 2.404113806, "cmrs_value": 3.000,
                    "iterations": it}
        for (J, why) in fails:
            if isinstance(why, str) and "m ~" in why:
                m_star = float(why.split("m ~")[1].split(")")[0])
                extra += [(J, m_star**2), (J, (m_star * 1.02)**2),
                          (J, (m_star * 0.98)**2)]
    raise RuntimeError("no certifiable functional")
