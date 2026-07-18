"""CHECK 2: the improved sum rule CMRS eq (42)-(44), heavy AND low side,
plus the J=0 closed form eq (56) and the general-D Gegenbauer Taylor rule.
"""
import sympy as sp

u, m2 = sp.symbols("u m2", positive=False), sp.Symbol("m2", positive=True)
u = sp.Symbol("u")
P1, P1p = sp.symbols("P1 P1p")   # P_J(1), P'_J(1)

# per-state kernel of C_k,u  (eq 23 heavy side):
def Bk(k, uu):
    # (2m^2+u)/(m^2+u) * P_J(1+2u/m^2) / [m^2(m^2+u)]^{k/2}
    # near u=0 only P1, P1p needed for value/derivative at 0
    PJ = P1 + P1p * 2 * uu / m2          # exact to first order at uu=0
    return (2 * m2 + uu) / (m2 + uu) * PJ / (m2 * (m2 + uu)) ** (k // 2)

# B_{2n}(0) and B'_{2n}(0)
n = sp.Symbol("n", integer=True, positive=True)
B0 = sp.simplify(Bk(2, sp.Integer(0)).subs(u, 0))
print("generic B_k(0) with k=2:", B0)   # expect 2*P1/m2^2

# closed forms derived by hand, verified for explicit n:
B2n_0 = 2 * P1 / m2 ** (2 * n)
B2n_p0 = ((-2 * n - 1) * P1 + 4 * P1p) / m2 ** (2 * n + 1)
for nn in (1, 2, 3, 4):
    uu = sp.Symbol("uu")
    expr = (2 * m2 + uu) / (m2 + uu) * (P1 + P1p * 2 * uu / m2) \
        / (m2 * (m2 + uu)) ** nn
    v0 = sp.simplify(expr.subs(uu, 0))
    v1 = sp.simplify(sp.diff(expr, uu).subs(uu, 0))
    assert sp.simplify(v0 - B2n_0.subs(n, nn)) == 0, nn
    assert sp.simplify(v1 - B2n_p0.subs(n, nn)) == 0, (nn, v1)
print("B_{2n}(0) = 2 P_J(1)/m^{4n},  B'_{2n}(0) = [-(2n+1)P_J(1)+4P'_J(1)]/m^{4n+2}: OK")

# eq (42) sum over n>=2, closed form:
w = u**2 / m2**2  # note m2 == m^2, so m^{4n} = m2^{2n}
S = sp.summation((n * u ** (2 * n - 2) * 2 * P1 / m2 ** (2 * n)
                  + u ** (2 * n - 1) * ((-2 * n - 1) * P1 + 4 * P1p)
                  / m2 ** (2 * n + 1)), (n, 2, sp.oo))
S = sp.simplify(S)
print("sum_{n>=2} [n u^{2n-2} C_{2n,0} + u^{2n-1} C'_{2n,0}] per state =")
print("  ", sp.factor(S))

# eq (44) subtraction term:  u^2/m^6 [ (4m^2+3u) P1 / (m^2+u)^2 + 4u P1p/(m^4-u^2) ]
sub44 = u**2 / m2**3 * ((4 * m2 + 3 * u) * P1 / (m2 + u) ** 2
                        + 4 * u * P1p / (m2**2 - u**2))
if isinstance(S, sp.Piecewise):
    S = S.args[0][0]           # convergent branch |u^2/m^4| < 1
diff = sp.simplify(S - sub44)
print("S - eq(44) subtraction term (convergent branch):", diff)
assert diff == 0, "eq (44) heavy-side identity FAILED"
print("eq (44) heavy kernel identity: OK  (valid for |u| < m^2; geometric series in u^2/m^4)")

# eq (56): J=0: P1=1, P1p=0, u=-p^2:
p = sp.Symbol("p", positive=True)
C2imp_J0 = ((2 * m2 + u) / (m2 * (m2 + u) ** 2) * 1
            - u**2 / m2**3 * ((4 * m2 + 3 * u) * 1 / (m2 + u) ** 2 + 0))
val = sp.simplify(C2imp_J0.subs(u, -p**2))
tgt = 2 / m2**2 + 3 * p**2 / m2**3
print("C2_imp[m^2, J=0](u=-p^2) =", sp.simplify(val), " target:", tgt)
assert sp.simplify(val - tgt) == 0, "eq (56) FAILED"
print("eq (56): OK  -> J=0 improved kernel is EXACTLY 2/m^4 + 3p^2/m^6 (positive, m-indep shape)")

# ---- low side of the improvement (42): with the verified C_k EFT expressions,
# C_2 - sum_{n=2,3} [n u^{2n-2} C_{2n,0} + u^{2n-1} C'_{2n,0}] must equal
# 8piG/(-u) + 2 g2 - g3 u  through the order-8 couplings included.
G8, g2, g3, g4, g5, g6, g6p, g7 = sp.symbols("G8 g2 g3 g4 g5 g6 g6p g7")
C2 = -G8 / u + 2 * g2 - g3 * u + 8 * g4 * u**2 - 2 * g5 * u**3 \
    + 24 * g6 * u**4 - 4 * g7 * u**5
C4 = 4 * g4 - 2 * g5 * u + (24 * g6 + g6p) * u**2 - 8 * g7 * u**3
C6 = 8 * g6 - 4 * g7 * u
C4_0, C4p_0 = C4.subs(u, 0), sp.diff(C4, u).subs(u, 0)
C6_0, C6p_0 = C6.subs(u, 0), sp.diff(C6, u).subs(u, 0)
C2imp_low = sp.expand(C2 - (2 * u**2 * C4_0 + u**3 * C4p_0)
                      - (3 * u**4 * C6_0 + u**5 * C6p_0))
print("C2_improved | EFT =", C2imp_low)
assert sp.simplify(C2imp_low - (-G8 / u + 2 * g2 - g3 * u)) == 0, \
    "eq (43) FAILED at this coupling order"
print("eq (42)->(43) low side: OK (g4..g7, g6' all cancelled; only G, g2, g3 survive)")

# ---- general-D Gegenbauer Taylor rule:
# P_J(x) = 2F1(-J, J+D-3, (D-2)/2, (1-x)/2)   (eq 9)
# claim: P_J(1+y) = sum_r (y/2)^r / (r! ((D-2)/2)_r) * prod_{i=0}^{r-1}(Jc2 - i(i+D-3))
# with Jc2 = J(J+D-3); check against explicit 2F1 for J=0,2,4,6, D symbolic.
D, y = sp.symbols("D y")
def PJ_explicit(J):
    z = -y / 2      # (1-x)/2 with x = 1+y
    out = 0
    for r in range(J + 1):
        out += (sp.rf(-J, r) * sp.rf(J + D - 3, r) / sp.rf((D - 2) / 2, r)
                / sp.factorial(r)) * z**r
    return sp.expand(out)
def PJ_taylor(J, order):
    Jc2 = J * (J + D - 3)
    out = 0
    for r in range(order + 1):
        prod = sp.Integer(1)
        for i in range(r):
            prod *= (Jc2 - i * (i + D - 3))
        out += (y / 2) ** r / (sp.factorial(r) * sp.rf((D - 2) / 2, r)) * prod
    return sp.expand(out)
for J in (0, 2, 4, 6):
    d = sp.simplify(PJ_explicit(J) - PJ_taylor(J, J))
    assert d == 0, (J, d)
print("general-D Taylor rule P_J(1+y) = sum_r (y/2)^r/(r! ((D-2)/2)_r) "
      "prod_{i<r}(J(J+D-3) - i(i+D-3)): OK")
# P'_J(1) = J(J+D-3)/(D-2):
for J in (2, 4, 6):
    pp = sp.simplify(sp.diff(PJ_explicit(J), y).subs(y, 0)
                     - J * (J + D - 3) / (D - 2))
    assert pp == 0, J
print("P'_J(1) = J(J+D-3)/(D-2): OK ;  P''_J(1) = Jc2(Jc2-(D-2))/(2(D-2)... check:")
for J in (2, 4, 6):
    Jc2 = J*(J+D-3)
    tgt = Jc2*(Jc2-(D-2))/(2*(D-2)*(D/2))   # (y/2)^2/(2! (c)_2) * Jc2(Jc2-(D-2)) *2 ... derive
    val = sp.simplify(sp.diff(PJ_explicit(J), y, 2).subs(y, 0)/2)
    print("  J=", J, " P''_J(1)/2 coeff:", sp.simplify(val), " formula:",
          sp.simplify(Jc2*(Jc2-(D-2))/(4*(D-2)*(D)/2/1)), " match:",
          sp.simplify(val - Jc2*(Jc2-(D-2))/(2*(D-2)*D)) == 0)
# D=4 reduction equals the code's _pj_taylor
X = sp.Symbol("X")
def pj_taylor_code(order):
    out = sp.Integer(0)
    for r in range(order + 1):
        prod = sp.Integer(1)
        for i in range(1, r + 1):
            prod *= (X - i * (i - 1))
        out += y**r * prod / (sp.Integer(2)**r * sp.factorial(r)**2)
    return out
for J in (0, 2, 4):
    d = sp.simplify(PJ_taylor(J, J).subs(D, 4)
                    - pj_taylor_code(J).subs(X, J * (J + 1)))
    assert d == 0, J
print("D=4 reduction matches positivity.py _pj_taylor: OK")
