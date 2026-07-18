"""CHECK 4b: mapping X_k,u to known null rows; single-state consistency;
eq (86) numeric; end-to-end single-scalar model test of eq (44).
"""
import sympy as sp
import mpmath as mp

D = sp.Symbol("D", positive=True)
u, up = sp.symbols("u u'")
m2 = sp.Symbol("m2", positive=True)

def PJ(J, x):
    z = (1 - x) / 2
    return sp.expand(sum(sp.rf(-J, r) * sp.rf(J + D - 3, r)
                         / sp.rf((D - 2) / 2, r) / sp.factorial(r) * z**r
                         for r in range(J + 1)))

def Xk(J, k, uu):
    first = ((2 * m2 + uu) / (uu * m2 * (m2 + uu))
             * m2 * PJ(J, 1 + 2 * uu / m2) / (uu * m2 * (m2 + uu)) ** (k // 2))
    integrand = ((2 * m2 + up) * (m2 - up) * (m2 + 2 * up)
                 / (m2 * (uu - up) * up * (m2 - uu) * (m2 + up) * (m2 + uu + up))
                 * m2 * PJ(J, 1 + 2 * up / m2)
                 / (up * m2 * (m2 + up)) ** (k // 2))
    second = sp.residue(integrand, up, 0)
    return sp.simplify(first - second)

# (a) X_2,0 vs eq (27) row  J^2(2J^2-5D+4)/m^8
print("--- X_2,u->0 limit vs eq (27) ---")
for J in (0, 2, 4):
    x20 = sp.simplify(sp.limit(Xk(J, 2, u), u, 0))
    Jc2 = J * (J + D - 3)
    tgt = Jc2 * (2 * Jc2 - 5 * D + 4)
    if tgt == 0:
        print(f"  J={J}: X_2,0 = {x20} (target row = 0)")
    else:
        print(f"  J={J}: X_2,0 * m^? / row =", sp.simplify(sp.factor(x20 / tgt)))

# (b) single heavy scalar consistency: X_k,u[m^2, J=0] == 0 identically?
for k in (2, 4, 6):
    val = sp.simplify(Xk(0, k, u))
    print(f"X_{k},u[m^2, J=0] =", val)

# (c) end-to-end: M_model = g^2 [1/(m0^2-s)+1/(m0^2-t)+1/(m0^2-u)]
#     low side: compute C_k,u|EFT by residue on the EFT (Taylor) expansion,
#     build C_2^improved via eq (42) truncated, compare with
#     c * C2imp_kernel[m0^2, 0](u), c = g^2/m0^2 (heavy-average weight).
s = sp.Symbol("s")
g2c, m02 = sp.symbols("gsq m02", positive=True)
t = -s - u
NORD = 8  # keep EFT couplings through total Mandelstam degree 8
def expand_pole(x):  # 1/(m0^2 - x) Taylor to degree NORD
    return sum(x**k / m02 ** (k + 1) for k in range(NORD + 1))
M_eft = g2c * (expand_pole(s) + expand_pole(t) + expand_pole(u))
def Ck_eft(expr, k):
    integ = (2 * s + u) / (s * (s + u)) * expr / (s * (s + u)) ** (k // 2)
    return sp.expand(sp.residue(integ, s, 0))
C2 = Ck_eft(M_eft, 2)
subs_terms = 0
for n in (2, 3, 4):
    C2n = Ck_eft(M_eft, 2 * n)
    C2n_0 = C2n.subs(u, 0)
    C2n_p0 = sp.diff(C2n, u).subs(u, 0)
    subs_terms += n * u ** (2 * n - 2) * C2n_0 + u ** (2 * n - 1) * C2n_p0
C2imp_low = sp.expand(C2 - subs_terms)
# heavy side: c * (2/m0^4 - 3u/m0^6), c = g^2/m0^2   [eq (56) with u=-p^2]
heavy = g2c / m02 * (2 / m02**2 - 3 * u / m02**3)
diff = sp.simplify(sp.series(C2imp_low - heavy, u, 0, 6).removeO())
print("end-to-end single-scalar: C2imp_low - c*kernel[m0^2,0] =", diff,
      " (should vanish through u^5 given degree-8 EFT truncation)")

# (d) eq (86) numeric: D=7, n=2, b=3.7
mp.mp.dps = 25
for (Dv, nn, bv) in [(7, 2, 3.7), (6, 1.5, 2.2), (10, 3.5, 8.0)]:
    bv = mp.mpf(bv)
    nu = (Dv - 4) / mp.mpf(2)
    lhs = mp.gamma((Dv - 2) / mp.mpf(2)) * mp.quad(
        lambda p: p**mp.mpf(nn) * mp.besselj(nu, bv * p) / (bv * p / 2) ** nu,
        [0, 1])
    rhs = mp.hyper([(mp.mpf(nn) + 1) / 2],
                   [(Dv - 2) / mp.mpf(2), (mp.mpf(nn) + 3) / 2],
                   -bv**2 / 4) / (mp.mpf(nn) + 1)
    print(f"eq (86) D={Dv} n={nn} b={bv}: lhs={mp.nstr(lhs,15)} rhs={mp.nstr(rhs,15)} "
          f"diff={mp.nstr(lhs-rhs,3)}")
