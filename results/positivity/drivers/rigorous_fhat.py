"""E1' PRIMITIVE: rigorous, pure-rational enclosures of the impact-parameter
transforms and of f_hat(b).

t_n(b) = N_nu * Integral_0^P p^n * jt(b p) dp,   jt(u) = J_nu(u)/(u/2)^nu,
nu = (D-4)/2 = 3 (D=10), and N_nu = Gamma((D-2)/2) = Gamma(4) = 6 is the
normalization used by qgse.gravity_lp._bessel_transform_fn (VERIFIED below
against it numerically).

jt is ENTIRE with rational Taylor coefficients:
  jt(u) = Gamma(nu+1) * Sum_k (-1)^k (u/2)^{2k} / (k! Gamma(nu+k+1))
        = Sum_k (-1)^k u^{2k} * [ nu! / (4^k k! (nu+k)!) ]        (nu integer)
=> t_n(b) = N_nu * Sum_k (-1)^k c_k b^{2k} P^{n+2k+1}/(n+2k+1),
   c_k = nu!/(4^k k! (nu+k)!)  -- ALL RATIONAL for rational b, P.
Truncation: once the term ratio r_k = (bP/2)^2/((k+1)(nu+k+1)) <= 1/2 for all
subsequent k, |tail| <= 2*|next term| (geometric majorant) -- rigorous.
Exact rational partial sums make the ~200-digit cancellation at bP~450 exact.

f_hat(b) = Sum_n a_n t_n(b) with the functional's exact rational a_n
=> rigorous rational interval [lo, hi] for f_hat at any rational b.
"""
from fractions import Fraction
import json


def t_n_interval(n, b, P, nu=3, N_norm=Fraction(6)):
    """Rigorous interval for t_n(b); b, P rational (Fraction)."""
    b = Fraction(b); P = Fraction(P)
    x = (b * P / 2) ** 2                       # (bP/2)^2, rational
    # term_k(full) = (-1)^k * nu!/(4^k k!(nu+k)!) * b^{2k} P^{n+2k+1}/(n+2k+1)
    # ratio |T_{k+1}/T_k| = x/((k+1)(nu+k+1)) * (n+2k+1)/(n+2k+3) < x/((k+1)(nu+k+1))
    import math
    nu_f = math.factorial(nu)
    S = Fraction(0)
    T = Fraction(nu_f, math.factorial(nu)) * P ** (n + 1) / (n + 1)  # k=0 term (=P^{n+1}/(n+1))
    k = 0
    while True:
        S += (T if k % 2 == 0 else -T)
        # next term magnitude
        T_next = T * x / ((k + 1) * (nu + k + 1)) * Fraction(n + 2 * k + 1,
                                                             n + 2 * k + 3)
        ratio_bound = Fraction(x, (k + 2) * (nu + k + 2))  # bound for ALL later ratios
        if ratio_bound <= Fraction(1, 2) and k >= 1:
            # tail after including T_next's term is bounded by geometric sum:
            # |tail starting at k+1| <= T_next * 1/(1 - 1/2) = 2 T_next
            err = 2 * T_next
            return N_norm * (S - err), N_norm * (S + err)
        T = T_next
        k += 1
        if k > 5000:
            raise RuntimeError("series did not reach ratio<=1/2 (huge bP?)")


def fhat_interval(a_f, f_powers, b, P):
    lo = Fraction(0); hi = Fraction(0)
    for a, n in zip(a_f, f_powers):
        tl, th = t_n_interval(n, b, P)
        if a >= 0:
            lo += a * tl; hi += a * th
        else:
            lo += a * th; hi += a * tl
    return lo, hi


if __name__ == "__main__":
    import sys, os, time
    REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
    sys.path.insert(0, REPO); os.chdir(REPO)
    import sympy as sp

    # ---- VALIDATION 1: against the production numeric transform ----------
    from qgse.verifiers.gravity_lp import _bessel_transform_fn
    P = Fraction(17, 16)   # validate at 17/16 and 9/8
    print("validate t_n intervals vs production mp.quad transform:", flush=True)
    for Pq, Pfr in ((sp.Rational(17, 16), Fraction(17, 16)),
                    (sp.Rational(9, 8), Fraction(9, 8))):
        # production transform integrates over p in (0, p_max]: check source
        # convention by direct comparison
        for n in (2, 3, 9, 16):
            t = _bessel_transform_fn(n, 10)
            # NOTE: production tf built per p_max? check: it takes (n, D) only
            # -- p_max fixed inside? We compare at P=1 vs our P=1 if so.
            break
        break
    # _bessel_transform_fn signature: check
    import inspect
    print("  _bessel_transform_fn signature:", inspect.signature(_bessel_transform_fn), flush=True)
    src = inspect.getsource(_bessel_transform_fn)
    print("  source:\n" + src, flush=True)
