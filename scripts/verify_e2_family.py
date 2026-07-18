"""E_2 null-kernel family for the D=10 susy dispersive pipeline.

DERIVATION (recovered from CMRS 2102.08951, Sec 3.8 'Maximal supergravity',
LaTeX source sections/sec_localized_b.tex lines 405-417 -- the 'eq 75 family'):

The s<->u symmetry of the UNSUBTRACTED fixed-u dispersion relation (valid
under the susy Regge axiom s^2 M -> 0) gives the master crossing identity

    < K(s,u;m^2,J) >  =  < K(u,s;m^2,J) >        (s, u < 0)
    K(s,u) = m^2 (2m^2+u) P_J(1+2u/m^2) / ((m^2-s)(m^2+s+u))

whose EFT (low-energy) side is EXACTLY ZERO: the only massless singularity,
the graviton pole 8piG/(stu), is itself s<->u symmetric and cancels between
the two sides; every contact (g_0, g_2, ...) is *encoded in* the heavy
average (the kernel is the full dispersive representation of M - graviton:
K partial-fractions to m^2 P_J [1/(m^2-s) + 1/(m^2+s+u)]).

Hence every s-derivative at s=0 is an exact null:
    E_n[m^2,J](u) := d^n/ds^n [ K(s,u) - K(u,s) ] |_{s=0},  <E_n> = 0.
    E_0 = paper's null family (2m^2+u)P_J/(m^2+u) - 2m^4/(m^4-u^2)
          (== C_0^imp kernel - 2, since g_0 = <2>),
    E_1 = the repo's e1_expr   (verified exactly below),
    E_2 = the new family derived here.

This script is the executable validation battery. Run:
  source .venv/bin/activate && python e2_family.py
"""
import sys
import functools

import numpy as np
import sympy as sp

sys.path.insert(0, "/Users/mruckman1/Desktop/dev/quantum_gravity")

from qgse.verifiers.gravity_positivity import pj_poly, _M2, _U, _P
from qgse.verifiers.gravity_susy import (e1_expr, c0imp_expr, cm2_expr,
                                         c2imp_susy_expr, D)
from qgse.verifiers.gravity_lp import _xk_expr

_S = sp.Symbol("s")
FAILED = []


def ck(name, cond):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------- the family
def K_master(sv, uv, J):
    """Heavy kernel of the unsubtracted fixed-u dispersion relation."""
    pj = pj_poly(J, D, 1 + 2 * uv / _M2)
    return _M2 * (2 * _M2 + uv) * pj / ((_M2 - sv) * (_M2 + sv + uv))


def En_direct_u(n, J):
    """E_n by direct differentiation of the master identity (u-form)."""
    N = K_master(_S, _U, J) - K_master(_U, _S, J)
    return sp.diff(N, _S, n).subs(_S, 0)


@functools.lru_cache(maxsize=None)
def e2_expr(J: int):
    """E_2 null kernel (second s-derivative at s=0 of the master crossing
    identity, CMRS eq 75 family; E_1 = first derivative): exact null for
    susy-sector heavy densities (same s^2 M -> 0 Regge axiom)."""
    pj = pj_poly(J, D, 1 + 2 * _U / _M2)
    PJp1 = sp.Rational(J * (J + 7), 8)                          # P'_J(1), D=10
    PJpp1 = sp.Rational(J * (J + 7) * (J * (J + 7) - 8), 80)    # P''_J(1)
    E2 = (2 * _M2 * pj * (1 / _M2**3 + 1 / (_M2 + _U)**3)
          - _M2 / (_M2 - _U) * (4 * (PJp1 + 2 * PJpp1) / (_M2 * (_M2 + _U))
                                - 2 * (1 + 4 * PJp1) / (_M2 + _U)**2
                                + 4 * _M2 / (_M2 + _U)**3))
    return sp.cancel(sp.together(E2.subs(_U, -_P**2)))


# ---------------------------------------------------- (0) structural anchors
print("== (0) structural anchors of the master identity ==")
J = 4
pj = pj_poly(J, D, 1 + 2 * _U / _M2)
ck("K(s,u) == m^2 P_J(1+2u/m^2) [1/(m^2-s) + 1/(m^2+s+u)]  (dispersive rep)",
   sp.simplify(K_master(_S, _U, J)
               - _M2 * pj * (1 / (_M2 - _S) + 1 / (_M2 + _S + _U))) == 0)
t = -_S - _U
ck("graviton pole 1/(stu) is s<->u symmetric (cancels between the two sides)",
   sp.simplify((1 / (_S * t * _U)) - (1 / (_U * t * _S)).subs(
       {_S: _U, _U: _S}, simultaneous=True)) == 0)

# P'_J(1), P''_J(1) closed forms at D=10
x = sp.Symbol("x")
for Jt in (0, 2, 4, 8, 12):
    pjx = pj_poly(Jt, D, x)
    ok1 = sp.diff(pjx, x).subs(x, 1) == sp.Rational(Jt * (Jt + 7), 8)
    Jc = Jt * (Jt + 7)
    ok2 = sp.diff(pjx, x, 2).subs(x, 1) == sp.Rational(Jc * (Jc - 8), 80)
    ck(f"P'_J(1)=J(J+7)/8, P''_J(1)=Jc(Jc-8)/80 at J={Jt} (D=10)", ok1 and ok2)

# ------------------------------------- (1) E_0 ties the family to C_0^imp
print("== (1) E_0 == C_0^imp kernel - 2  (certified column anchor) ==")
for Jt in (0, 2, 4, 6):
    e0 = sp.cancel(sp.together(En_direct_u(0, Jt).subs(_U, -_P**2)))
    ck(f"E_0[J={Jt}] == c0imp_expr - 2",
       sp.simplify(e0 - (c0imp_expr(Jt) - 2)) == 0)

# --------------------------- (2) E_1 derivation recovered EXACTLY (the repo's)
print("== (2) e1_expr == d/ds[K(s,u)-K(u,s)]|_{s=0}  (derivation recovery) ==")
for Jt in (0, 2, 4, 6, 8, 10, 12):
    d1 = sp.cancel(sp.together(En_direct_u(1, Jt).subs(_U, -_P**2)))
    ck(f"E_1[J={Jt}] direct-diff == repo e1_expr",
       sp.simplify(d1 - e1_expr(Jt)) == 0)

# ------------------------------- (3) E_2 closed form == direct second derivative
print("== (3) e2_expr == d^2/ds^2[K(s,u)-K(u,s)]|_{s=0} ==")
for Jt in (0, 2, 4, 6, 8, 10, 12):
    d2 = sp.cancel(sp.together(En_direct_u(2, Jt).subs(_U, -_P**2)))
    ck(f"E_2[J={Jt}] direct-diff == closed-form e2_expr",
       sp.simplify(d2 - e2_expr(Jt)) == 0)

# --------------------------------- (4) corner regularity at u=-m^2 (p^2=m^2)
print("== (4) regularity at u=-m^2 for even J (finite limit) ==")
eps = sp.Symbol("eps", positive=True)
for name, fam in (("E_1", lambda Jt: En_direct_u(1, Jt)),
                  ("E_2", lambda Jt: En_direct_u(2, Jt))):
    for Jt in (0, 2, 4):
        ku = sp.cancel(sp.together(fam(Jt)))
        den = sp.denom(ku)
        reg = sp.simplify(den.subs(_U, -_M2)) != 0
        if reg:
            val = sp.simplify(ku.subs(_U, -_M2))
            ck(f"{name}[J={Jt}] regular at u=-m^2; limit = {val}", True)
        else:  # fall back to series limit
            ser = sp.series(ku.subs(_U, -_M2 + eps), eps, 0, 1).removeO()
            pole = any(sp.degree(sp.denom(sp.together(term)), eps) > 0
                       for term in sp.Add.make_args(sp.expand(ser)))
            ck(f"{name}[J={Jt}] finite limit at u=-m^2", not pole)
# audit-class check mirroring battery item 11 (pole order < 4 at the corner)
for Jt in (2, 6):
    kJ = sp.together(En_direct_u(2, Jt))
    ck(f"E_2[J={Jt}] pole-order<4 at u=-m^2 (audit class, battery-11 mirror)",
       sp.simplify(sp.limit(kJ * (_M2 + _U)**4, _U, -_M2)) == 0)

# ------------------------------------------------- (5) numeric sanity grid
print("== (5) numeric sanity: finite on (m2,p) grid, J=0..12 ==")
m2g = np.concatenate([np.linspace(1.0, 4.0, 13), np.geomspace(4.0, 400.0, 12)])
pg = np.linspace(0.01, 0.999, 21)
M2G, PG = np.meshgrid(m2g, pg)
all_ok, worst = True, 0.0
for Jt in range(0, 13, 2):
    fn = sp.lambdify((_M2, _P), e2_expr(Jt), "numpy")
    vals = np.asarray(fn(M2G, PG), dtype=float)
    finite = np.all(np.isfinite(vals))
    all_ok &= finite
    worst = max(worst, float(np.max(np.abs(vals))))
ck(f"E_2 finite on {M2G.size}-pt grid x J=0..12 (max |val| = {worst:.3e})",
   all_ok)

# ------------------------------------- (6) linear independence (rank check)
print("== (6) linear independence from existing columns ==")
COLS = [("C_-2", cm2_expr), ("C_0^imp", c0imp_expr),
        ("X_2", lambda J: _xk_expr(2, J, D)),
        ("X_4", lambda J: _xk_expr(4, J, D)),
        ("X_6", lambda J: _xk_expr(6, J, D)),
        ("C_2^imp-susy", c2imp_susy_expr), ("E_1", e1_expr), ("E_2", e2_expr)]
m2s = [1.3, 1.9, 2.7, 4.1, 6.3, 9.7, 15.1, 24.3]
ps = [0.15, 0.35, 0.55, 0.75, 0.95]
pts = [(a, b) for a in m2s for b in ps]


def col_vec(expr):
    fn = sp.lambdify((_M2, _P), expr, "numpy")
    return np.array([float(fn(a, b)) for (a, b) in pts])


stacked_with, stacked_without = [], []
for Jt in (2, 6, 10):
    mat = np.column_stack([col_vec(f(Jt)) for _, f in COLS])
    mat = mat / (np.abs(mat).max(axis=0) + 1e-300)      # column-normalize
    r_with = np.linalg.matrix_rank(mat, tol=1e-8)
    r_wo = np.linalg.matrix_rank(mat[:, :-1], tol=1e-8)
    sv = np.linalg.svd(mat, compute_uv=False)
    ck(f"J={Jt}: rank {r_wo}->{r_with} adding E_2 "
       f"(min/max sv = {sv[-1]/sv[0]:.2e})", r_with == r_wo + 1)
    stacked_with.append(mat)
    stacked_without.append(mat[:, :-1])
matW = np.vstack(stacked_with)
r_with = np.linalg.matrix_rank(matW, tol=1e-8)
r_wo = np.linalg.matrix_rank(np.vstack(stacked_without), tol=1e-8)
ck(f"stacked J=2,6,10 (functional-level): rank {r_wo}->{r_with} adding E_2",
   r_with == r_wo + 1)

print()
print("SUMMARY:", "ALL PASS" if not FAILED else f"FAILURES: {FAILED}")
sys.exit(0 if not FAILED else 1)
