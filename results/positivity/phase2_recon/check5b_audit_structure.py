"""CHECK 5 (fast version): kernel regularity at u=-m^2 and audit integral shape."""
import sympy as sp

D = sp.Symbol("D", positive=True)
u = sp.Symbol("u")
m2 = sp.Symbol("m2", positive=True)
p = sp.Symbol("p", positive=True)
eps = sp.Symbol("eps")

def PJ(J, x):
    z = (1 - x) / 2
    return sp.expand(sum(sp.rf(-J, r) * sp.rf(J + D - 3, r)
                         / sp.rf((D - 2) / 2, r) / sp.factorial(r) * z**r
                         for r in range(J + 1)))

def C2imp(J, uu):
    P1 = PJ(J, sp.Integer(1))
    xx = sp.Symbol("_x_")
    P1p = sp.diff(PJ(J, xx), xx).subs(xx, 1)
    return ((2 * m2 + uu) * PJ(J, 1 + 2 * uu / m2) / (m2 * (m2 + uu) ** 2)
            - uu**2 / m2**3 * ((4 * m2 + 3 * uu) * P1 / (m2 + uu) ** 2
                               + 4 * uu * P1p / (m2**2 - uu**2)))

# regularity at u = -m2: substitute u = -m2 + eps, Laurent in eps
for J in (0, 2, 4, 6):
    K = C2imp(J, -m2 + eps)
    ser = sp.series(sp.together(K), eps, 0, 1)
    # extract eps^-2 and eps^-1 coefficients
    c2 = sp.simplify(ser.coeff(eps, -2))
    c1 = sp.simplify(ser.coeff(eps, -1))
    assert c2 == 0 and c1 == 0, (J, c2, c1)
    print(f"J={J}: C2imp regular at u=-m^2 (1/eps^2 and 1/eps coeffs = 0), D symbolic")

xx = sp.Symbol("_x_")
for J in (0, 2, 4, 6):
    assert sp.simplify(PJ(J, -1) - 1) == 0, J
    assert sp.simplify(sp.diff(PJ(J, xx), xx).subs(xx, -1)
                       + sp.diff(PJ(J, xx), xx).subs(xx, 1)) == 0, J
print("P_J(-1)=1, P'_J(-1)=-P'_J(1) (even J, D symbolic): OK")

# large-J dominant coefficient at u=-p^2:
big = sp.simplify((-u**2 / m2**3) * 4 * u / (m2**2 - u**2)).subs(u, -p**2)
print("P'_J(1)-coefficient in C2imp at u=-p^2:", sp.simplify(big),
      "  [= 4 p^6 / (m^2^3 (m^4-p^4)) > 0 for p^2<m^2]")

# audit integral shape: D=7, J=4, n=2 via partial fractions in q=p^2
q = sp.Symbol("q", positive=True)
ker = sp.together(C2imp(4, -q).subs(D, 7))
ker = sp.apart(sp.simplify(ker), q)
print("D=7 J=4 kernel apart in q=p^2:", ker)
I = sp.simplify(sp.integrate(sp.sqrt(q) * ker / (2 * sp.sqrt(q)), (q, 0, 1)))
# note: int_0^1 dp p^2 K(-p^2) = int_0^1 dq sqrt(q)*K(-q)/(2 sqrt(q)) * ... fix:
# p^2 K dp = q K * dp, dp = dq/(2 sqrt q) -> integrand = q K/(2 sqrt q) = sqrt(q) K/2
I = sp.simplify(sp.integrate(sp.sqrt(q) * ker / 2, (q, 0, 1)))
print("I_{n=2}(m2), D=7, J=4:", I)
