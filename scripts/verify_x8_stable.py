"""Verify the k=8 jet-subtracted stable-kernel tables (rho(8,r), G(8,r))
against the exact X_8 crossing null _xk_expr(8, J, 10).

DERIVATION (reconstructed from _xk_kernel, gravity_lp.py, CMRS eq 51):
  X_k = t1 - Res_{up=0}[ W_k(up) * P_J(1 + 2 up/m2) ],
  t1     = (2m2+u) m2 P_J(1+2u/m2) / (u m2 (m2+u))^{k/2+1},
  W_k(up)= (2m2+up)(m2-up)(m2+2up) m2
           / [ m2 (u-up) up (m2-u) (m2+up) (m2+u+up) (up m2 (m2+up))^{k/2} ].
Taylor-expanding P_J at x=1 (P_J^{(r)}(1) = _pr1(r,J) for D=10), the
residue truncates at r = k/2 and
  X_k[J](m2,u) = rho(k,0) P_J(1+2u/m2) - sum_{r=0}^{k/2} P_J^{(r)}(1) rho(k,r),
  rho(k,r) = (2/m2)^r / r! * [up^{-1-r} Laurent coeff of W_k at up=0],
with rho(k,0) doing double duty: t1 coefficient == r=0 residue coefficient
(verified below for k=2,4,6,8). The jet-regular kernels are
  G(k,r) = rho(k,0) (x-1)^r / r! - rho(k,r),  x-1 = 2u/m2, u = -p2,
finite at p2 -> 0 (each rho alone blows up like 1/u^{k/2+1-r}).

Checks (all exact-symbolic or mpmath dps=50 vs exact-rational reference):
  [0] Laurent extraction reproduces the k=2,4,6 code tables entry-by-entry
  [1] pasted G(8,r) == rho0 (x-1)^r/r! - rho(8,r) symbolically
  [2] pasted rho(8,r): X_8 identity symbolic, J = 2..12 even
  [3] numeric X_8 (rho form) vs _xk_expr(8,J,10), random pts, tol 1e-25
  [4] numeric G(8,r) vs the jet-subtraction difference, tol 1e-25
  [5] G(8,r) finite and -> analytic limit as p2 -> 0
  [6] jet branch end-to-end vs exact at large-J small-p points, tol 1e-25
  [7] dps=16 stress: unsubtracted form loses everything, jet branch doesn't
"""
import sys, os, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympy as sp
import mpmath as mp
from qgse.verifiers.gravity_positivity import pj_poly, _M2, _U, _P
from qgse.verifiers.gravity_lp import _xk_kernel, _xk_expr
from qgse.verifiers.gravity_susy import SusyR4Verifier

D = 10
mp.mp.dps = 50
random.seed(20260716)
m2s, us = _M2, _U
p2s = sp.Symbol("p2", positive=True)
up = sp.Symbol("up")
fails = []


# ==== the k=8 table entries as delivered for gravity_susy.py ===============
def _rho8(r, m2, u):
    if r == 0: return (2*m2+u)/(m2**4*u**5*(m2+u)**5)
    if r == 1: return -2*(2*m2**6-3*m2**5*u+2*m2**4*u**2-2*m2**3*u**3-31*m2**2*u**4-36*m2*u**5-12*u**6)/(m2**10*u**4*(u-m2)*(m2+u)**4)
    if r == 2: return -2*(2*m2**4-5*m2**3*u+7*m2**2*u**2+24*m2*u**3+12*u**4)/(m2**10*u**3*(u-m2)*(m2+u)**3)
    if r == 3: return 4*(-2*m2**2+7*m2*u+7*u**2)/(3*m2**10*u**2*(u-m2)*(m2+u)**2)
    if r == 4: return -4/(3*m2**10*u*(u-m2)*(m2+u))
    raise KeyError(r)


def _G8(r, m2, p2):
    if r == 1:
        return 2*(33*m2**3 - 67*m2**2*p2 + 48*m2*p2**2 - 12*p2**3)/(m2**10*(m2 - p2)**5*(m2 + p2))
    if r == 2:
        return -_G8(1, m2, p2)
    if r == 3:
        return 4*(21*m2**3 - 40*m2**2*p2 + 28*m2*p2**2 - 7*p2**3)/(3*m2**10*(m2 - p2)**5*(m2 + p2))
    if r == 4:
        return -2*(9*m2**3 - 13*m2**2*p2 + 8*m2*p2**2 - 2*p2**3)/(3*m2**10*(m2 - p2)**5*(m2 + p2))
    raise KeyError(r)
# ===========================================================================


def pr1_mp(r, J):
    out = mp.mpf(1)
    for i in range(r):
        out *= (J * (J + 7) - i * (i + 7)) / mp.mpf(2 * (4 + i))
    return out


def pr1_exact(r, J):
    out = sp.Integer(1)
    for i in range(r):
        out *= sp.Rational(J * (J + 7) - i * (i + 7), 2 * (4 + i))
    return out


def rho_derived(k, r):
    """the derivation: Laurent coefficient extraction from W_k."""
    R = k // 2
    W = ((2 * m2s + up) * (m2s - up) * (m2s + 2 * up)
         / (m2s * (us - up) * up * (m2s - us) * (m2s + up)
            * (m2s + us + up)) * m2s / (up * m2s * (m2s + up)) ** R)
    reg = sp.cancel(sp.together(W * up ** (R + 1)))
    w = sp.expand(sp.series(reg, up, 0, R + 1).removeO()).coeff(up, R - r)
    return sp.cancel(sp.Integer(2) ** r / (m2s ** r * sp.factorial(r)) * w)


def exact_ref(J, m2r, pr):
    v = _xk_expr(8, J, D).subs({_M2: m2r, _P: pr})
    assert v.is_Rational
    return mp.mpf(int(v.p)) / mp.mpf(int(v.q))


def x8_stable(J, m2, u, tail_eps_dps=None):
    """_xk_stable specialized to k=8 with the delivered tables."""
    R, s_ = 4, -2 * u / m2
    eps = mp.mpf(10) ** (-(tail_eps_dps or 50))
    if J * (J + 7) * s_ < 20:                       # jet branch
        tot = mp.mpf(0)
        for r in range(1, R + 1):
            tot += pr1_mp(r, J) * _G8(r, m2, -u)
        r0, xm1 = _rho8(0, m2, u), 2 * u / m2
        for r in range(R + 1, 81):
            t = pr1_mp(r, J) * r0 * xm1**r / mp.factorial(r)
            tot += t
            if abs(t) < eps * (1 + abs(tot)):
                break
        return tot
    Pj = mp.hyp2f1(-J, J + 7, mp.mpf(4), -u / m2)   # else branch
    return _rho8(0, m2, u) * Pj - sum(
        pr1_mp(r, J) * _rho8(r, m2, u) for r in range(R + 1))


def x8_unsubtracted(J, m2, u):
    Pj = mp.hyp2f1(-J, J + 7, mp.mpf(4), -u / m2)
    return _rho8(0, m2, u) * Pj - sum(
        pr1_mp(r, J) * _rho8(r, m2, u) for r in range(5))


# ---- [0] the derivation reproduces the k=2,4,6 tables + double duty + k=8
print("[0] Laurent extraction vs code tables (k=2,4,6) and pasted k=8:")
ok = True
for k in (2, 4, 6):
    for r in range(k // 2 + 1):
        ok &= sp.cancel(sp.together(
            rho_derived(k, r) - SusyR4Verifier._rho(k, r, m2s, us))) == 0
for r in range(5):
    ok &= sp.cancel(sp.together(rho_derived(8, r) - _rho8(r, m2s, us))) == 0
dd = all(sp.cancel((2*m2s+us)*m2s/(us*m2s*(m2s+us))**(k//2+1)
                   - rho_derived(k, 0)) == 0 for k in (2, 4, 6, 8))
ok &= dd
fails += [] if ok else ["derivation reconstruction"]
print(f"    all rho tables from Laurent coeffs + rho0 double duty: "
      f"{'PASS' if ok else 'FAIL'}")

# ---- [1] symbolic: pasted G(8,r) == rho(8,0)*(x-1)^r/r! - rho(8,r)
print("[1] symbolic: G(8,r) == rho0*(x-1)^r/r! - rho_r  (u = -p2):")
for r in range(1, 5):
    rhs = (_rho8(0, m2s, us) * (2 * us / m2s)**r / sp.factorial(r)
           - _rho8(r, m2s, us)).subs(us, -p2s)
    ok = sp.cancel(sp.together(_G8(r, m2s, p2s) - rhs)) == 0
    fails += [] if ok else [f"G(8,{r}) symbolic"]
    print(f"    r={r}: {'PASS' if ok else 'FAIL'}")

# ---- [2] symbolic: X_8 identity with pasted rho table
print("[2] symbolic: X_8[J] == rho0*P_J - sum_r pr1(r)*rho(8,r):")
for J in (2, 4, 6, 8, 10, 12):
    pj = pj_poly(J, D, 1 + 2 * us / m2s)
    rhs = _rho8(0, m2s, us) * pj - sum(
        pr1_exact(r, J) * _rho8(r, m2s, us) for r in range(5))
    ok = sp.cancel(sp.together(_xk_kernel(8, J, D) - rhs)) == 0
    fails += [] if ok else [f"X8 symbolic J={J}"]
    print(f"    J={J:2d}: {'PASS' if ok else 'FAIL'}")

# ---- [3] numeric: rho form vs exact _xk_expr(8,J,10)
print("[3] numeric X_8 vs exact rational, 30 random pts, dps=50, tol 1e-25:")
worst = mp.mpf(0)
for _ in range(30):
    J = random.choice([4, 6, 8, 10, 12, 14, 16, 18, 20])
    m2r = sp.Rational(random.randint(15, 400), 10)     # 1.5 .. 40
    pr = sp.Rational(random.randint(10, 110), 100)     # 0.1 .. 1.1
    ex = exact_ref(J, m2r, pr)
    got = x8_stable(J, mp.mpf(m2r.p) / mp.mpf(m2r.q),
                    -(mp.mpf(pr.p) / mp.mpf(pr.q))**2)
    worst = max(worst, abs(got - ex) / (1 + abs(ex)))
ok = worst < mp.mpf("1e-25")
fails += [] if ok else ["X8 numeric"]
print(f"    worst rel err: {mp.nstr(worst, 3)}  {'PASS' if ok else 'FAIL'}")

# ---- [4] numeric: G(8,r) == jet-subtraction difference
print("[4] numeric G(8,r) vs rho0*(x-1)^r/r! - rho_r, tol 1e-25:")
worst = mp.mpf(0)
for _ in range(20):
    m2 = mp.mpf(random.randint(15, 400)) / 10
    p2 = (mp.mpf(random.randint(10, 110)) / 100)**2
    for r in range(1, 5):
        ref = (_rho8(0, m2, -p2) * (-2 * p2 / m2)**r / mp.factorial(r)
               - _rho8(r, m2, -p2))
        worst = max(worst, abs(_G8(r, m2, p2) - ref) / (1 + abs(ref)))
ok = worst < mp.mpf("1e-25")
fails += [] if ok else ["G8 numeric"]
print(f"    worst rel err: {mp.nstr(worst, 3)}  {'PASS' if ok else 'FAIL'}")

# ---- [5] regularity as p2 -> 0
print("[5] G(8,r) regular at p2 -> 0 (limits (66,-66,28,-6)/m2^13):")
lims = {1: 66, 2: -66, 3: 28, 4: -6}
ok = True
for m2 in (mp.mpf("1.5"), mp.mpf(5), mp.mpf(40)):
    for r in range(1, 5):
        vals = [_G8(r, m2, mp.mpf(10)**(-e)) for e in (4, 8, 16, 30)]
        lim = lims[r] / m2**13
        ok &= all(mp.isfinite(v) for v in vals)
        ok &= max(abs(v - lim) / abs(lim) for v in vals[-2:]) < mp.mpf("1e-14")
fails += [] if ok else ["G8 regularity"]
print(f"    finite + converge to limit, m2 in (1.5,5,40): "
      f"{'PASS' if ok else 'FAIL'}")

# ---- [6] jet branch end-to-end vs exact
print("[6] jet branch vs exact at jet-regime points, tol 1e-25:")
worst = mp.mpf(0)
for (J, m2r, pr) in [(40, sp.Rational(4), sp.Rational(1, 10)),
                     (40, sp.Rational(4), sp.Rational(1, 50)),
                     (40, sp.Rational(3, 2), sp.Rational(1, 20)),
                     (60, sp.Rational(10), sp.Rational(1, 10)),
                     (60, sp.Rational(3, 2), sp.Rational(1, 100))]:
    m2 = mp.mpf(m2r.p) / mp.mpf(m2r.q)
    u = -(mp.mpf(pr.p) / mp.mpf(pr.q))**2
    assert J * (J + 7) * (-2 * u / m2) < 20
    ex = exact_ref(J, m2r, pr)
    rel = abs(x8_stable(J, m2, u) - ex) / (1 + abs(ex))
    worst = max(worst, rel)
    print(f"    J={J} m2={float(m2r):5.2f} p={float(pr):5.3f}: "
          f"rel={mp.nstr(rel, 3)}")
ok = worst < mp.mpf("1e-25")
fails += [] if ok else ["X8 jet branch"]
print(f"    worst rel err: {mp.nstr(worst, 3)}  {'PASS' if ok else 'FAIL'}")

# ---- [7] the point of the exercise: dps=16 stress at small p
print("[7] dps=16 stress (unsubtracted rho form vs jet branch):")
for (J, m2r, pr) in [(60, sp.Rational(3, 2), sp.Rational(1, 1000)),
                     (100, sp.Rational(3, 2), sp.Rational(1, 1000))]:
    m2 = mp.mpf(m2r.p) / mp.mpf(m2r.q)
    u = -(mp.mpf(pr.p) / mp.mpf(pr.q))**2
    ref = x8_stable(J, m2, u)                       # dps 50
    mp.mp.dps = 16
    m2 = mp.mpf(m2r.p) / mp.mpf(m2r.q)
    u = -(mp.mpf(pr.p) / mp.mpf(pr.q))**2
    eu = x8_unsubtracted(J, m2, u)
    ej = x8_stable(J, m2, u, tail_eps_dps=16)
    mp.mp.dps = 50
    print(f"    J={J} m2={float(m2r)} p={float(pr)}: "
          f"unsubtracted rel={mp.nstr(abs(mp.mpf(eu)-ref)/abs(ref), 3)}, "
          f"jet rel={mp.nstr(abs(mp.mpf(ej)-ref)/abs(ref), 3)}")

print("\n==== SUMMARY:", "ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(0 if not fails else 1)
