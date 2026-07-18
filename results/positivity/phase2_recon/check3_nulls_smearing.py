"""CHECK 3: X_k null kernels (eq 51): identity eq (45), forward limit vs eq (27);
CHECK 4: smeared integrals eq (83) numerically + closed-form structure + endpoint;
CHECK 5: Bessel limit eq (48) numerically.
"""
import sympy as sp
import mpmath as mp

D = sp.Symbol("D", positive=True)
u, up, m2 = sp.symbols("u u' m2")
m2 = sp.Symbol("m2", positive=True)

def PJ(J, x):
    """Gegenbauer 𝒫_J(x) = 2F1(-J, J+D-3, (D-2)/2, (1-x)/2), exact poly."""
    z = (1 - x) / 2
    out = 0
    for r in range(J + 1):
        out += (sp.rf(-J, r) * sp.rf(J + D - 3, r) / sp.rf((D - 2) / 2, r)
                / sp.factorial(r)) * z**r
    return sp.expand(out)

def C2imp(J, uu):
    """eq (44) bracket."""
    P1 = PJ(J, sp.Integer(1))
    x = sp.Symbol("_x_")
    P1p = sp.diff(PJ(J, x), x).subs(x, 1)
    return ((2 * m2 + uu) * PJ(J, 1 + 2 * uu / m2) / (m2 * (m2 + uu) ** 2)
            - uu**2 / m2**3 * ((4 * m2 + 3 * uu) * P1 / (m2 + uu) ** 2
                               + 4 * uu * P1p / (m2**2 - uu**2)))

def Xk(J, k, uu):
    """eq (51)."""
    first = ((2 * m2 + uu) / (uu * m2 * (m2 + uu))
             * m2 * PJ(J, 1 + 2 * uu / m2) / (uu * m2 * (m2 + uu)) ** (k // 2))
    integrand = ((2 * m2 + up) * (m2 - up) * (m2 + 2 * up)
                 / (m2 * (uu - up) * up * (m2 - uu) * (m2 + up) * (m2 + uu + up))
                 * m2 * PJ(J, 1 + 2 * up / m2)
                 / (up * m2 * (m2 + up)) ** (k // 2))
    second = sp.residue(integrand, up, 0)
    return sp.simplify(first - second)

# ---- eq (45): C2imp(u) = C2(0) + u C2'(0) + u^2 X_{2,u}  (per state, no gravity)
for J in (0, 2):
    B2 = (2 * m2 + u) / (m2 + u) * PJ(J, 1 + 2 * u / m2) / (m2 * (m2 + u))
    C20 = B2.subs(u, 0)
    C20p = sp.diff(B2, u).subs(u, 0)
    lhs = C2imp(J, u)
    rhs = C20 + u * C20p + u**2 * Xk(J, 2, u)
    d = sp.simplify(lhs - rhs)
    assert d == 0, (J, d)
    print(f"eq (45) with X_2 from eq (51): OK for J={J} (D symbolic)")

# ---- X_4 forward limit vs eq (27): expect X_{4,0} ∝ 𝒥²(2𝒥²-5D+4)/m^8
for J in (0, 2, 4):
    x40 = sp.simplify(sp.limit(Xk(J, 4, u), u, 0))
    Jc2 = J * (J + D - 3)
    tgt = Jc2 * (2 * Jc2 - 5 * D + 4)
    if J == 0:
        assert sp.simplify(x40) == 0
        print("X_4,0[J=0] = 0: OK")
    else:
        ratio = sp.simplify(x40 * m2**4 / tgt)
        print(f"X_4,0[J={J}] * m^8 / (Jc2(2Jc2-5D+4)) =", ratio)

# D=4 cross-check against positivity.py n4 = 2X(X-8)/m^8 (up to overall const)
x40_J2_D4 = sp.simplify(sp.limit(Xk(2, 4, u), u, 0).subs(D, 4))
n4_J2 = sp.Rational(2) * 6 * (6 - 8)   # 2X(X-8) at X=6
print("D=4, J=2: X_4,0 * m^8 =", sp.simplify(x40_J2_D4 * m2**4),
      " ; n4 row value =", n4_J2, " ratio:",
      sp.simplify(x40_J2_D4 * m2**4 / n4_J2))

# ---- eq (83) numeric check, D=7, J=2, several n, m2
mp.mp.dps = 30
Dv = 7
x_ = sp.Symbol("x_")
for (nn, m2v) in [(2, sp.Rational(3, 2)), (3, 2), (5, sp.Rational(9, 8))]:
    ker = C2imp(2, -x_**2).subs({D: Dv, m2: m2v})
    fker = sp.lambdify(x_, ker, "mpmath")
    lhs = mp.quad(lambda p: p**nn * fker(p), [0, 1])
    n_, m2n = nn, mp.mpf(str(sp.nsimplify(m2v).evalf(30)))
    rhs = (-4 * (Dv - 1) * mp.hyp2f1(1, (n_ + 1) / 2, (n_ + 3) / 2, -1 / m2n)
           / ((Dv - 2) * m2n**2 * (n_ + 1))
           + 2 * (3 * Dv - 4) / ((Dv - 2) * m2n**2 * (n_ + 1))
           - 3 * (3 * Dv - 2) / ((Dv - 2) * m2n**3 * (n_ + 3)))
    print(f"eq (83) D=7 J=2 n={nn} m2={m2v}: lhs={mp.nstr(lhs,20)} rhs={mp.nstr(rhs,20)}"
          f"  diff={mp.nstr(lhs-rhs,3)}")

# ---- endpoint behavior at m = M = 1: is the p-integral convergent?
for Dv2 in (5, 7, 10):
    ker = C2imp(2, -x_**2).subs({D: Dv2, m2: 1})
    # behavior as p -> 1
    ser = sp.series(sp.simplify(ker), x_, 1, 2)
    print(f"D={Dv2}: C2imp[m2=1,J=2](-p^2) near p=1:", ser)

# ---- closed form for integer n (audit design): sympy should give elementary fns
expr = sp.integrate(x_**2 * C2imp(2, -x_**2).subs(D, 7), (x_, 0, 1))
print("closed form n=2, J=2, D=7 (m2 symbolic):")
print(" ", sp.simplify(expr))

# ---- Bessel limit eq (48), numeric: D=6, b=3, p=0.7; J = m*b/2, m large
Dv = 6
b, pv = mp.mpf(3), mp.mpf("0.7")
x = sp.Symbol("x")
for mv in (40, 100, 200):
    J = int(round(mv * float(b) / 2))
    Jp = sp.Symbol("Jp")
    # numeric Gegenbauer via mpmath hyp2f1 (J integer, terminating)
    val = mp.hyp2f1(-J, J + Dv - 3, (Dv - 2) / mp.mpf(2),
                    (pv**2) / mp.mpf(mv) ** 2)
    tgt = (mp.gamma((Dv - 2) / mp.mpf(2)) / (b * pv / 2) ** ((Dv - 4) / mp.mpf(2))
           * mp.besselj((Dv - 4) / mp.mpf(2), b * pv))
    # note: argument 1 - 2p^2/m^2 -> z=(1-x)/2 = p^2/m^2; b = 2J/m exactly
    beff = 2 * J / mp.mpf(mv)
    tgt2 = (mp.gamma((Dv - 2) / mp.mpf(2)) / (beff * pv / 2) ** ((Dv - 4) / mp.mpf(2))
            * mp.besselj((Dv - 4) / mp.mpf(2), beff * pv))
    print(f"m={mv} J={J}: P_J(1-2p^2/m^2)={mp.nstr(val,10)}  Bessel(b=2J/m)={mp.nstr(tgt2,10)}")
