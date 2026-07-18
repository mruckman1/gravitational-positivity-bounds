"""CHECK 5: structure needed by the exact a-posteriori audit.
(a) C_2imp[m^2,J](u) has NO pole at u=-m^2 (even J): residue and double-pole
    coefficient vanish identically in D  =>  smeared integrals contain only
    rational + atan pieces (from m^2+p^2), never atanh/log(m^2-p^2).
(b) P_J(-1)=1, P'_J(-1)=-P'_J(1) for even J.
(c) large-J at fixed m: kernel = P'_J(1)-term dominant, coefficient
    +4p^6/(m^6(m^4-p^4)) * J^2/(D-2)  -> sign control object T(m).
(d) closed-form integrals for J=4 (audit shape).
"""
import sympy as sp

D = sp.Symbol("D", positive=True)
u = sp.Symbol("u")
m2 = sp.Symbol("m2", positive=True)
p = sp.Symbol("p", positive=True)

def PJ(J, x):
    z = (1 - x) / 2
    return sp.expand(sum(sp.rf(-J, r) * sp.rf(J + D - 3, r)
                         / sp.rf((D - 2) / 2, r) / sp.factorial(r) * z**r
                         for r in range(J + 1)))

def C2imp(J, uu):
    P1 = PJ(J, sp.Integer(1))
    x = sp.Symbol("_x_")
    P1p = sp.diff(PJ(J, x), x).subs(x, 1)
    return ((2 * m2 + uu) * PJ(J, 1 + 2 * uu / m2) / (m2 * (m2 + uu) ** 2)
            - uu**2 / m2**3 * ((4 * m2 + 3 * uu) * P1 / (m2 + uu) ** 2
                               + 4 * uu * P1p / (m2**2 - uu**2)))

for J in (0, 2, 4, 6):
    K = sp.together(C2imp(J, u))
    res = sp.residue(K, u, -m2)
    # double pole coefficient: limit (u+m2)^2 K
    dp_ = sp.simplify(sp.limit((u + m2) ** 2 * K, u, -m2))
    assert sp.simplify(res) == 0, (J, res)
    assert dp_ == 0, (J, dp_)
    print(f"J={J}: C2imp regular at u=-m^2 (residue=0, double-pole coeff=0), D symbolic")

x = sp.Symbol("x")
for J in (0, 2, 4, 6):
    assert sp.simplify(PJ(J, -1) - 1) == 0, J
    assert sp.simplify(sp.diff(PJ(J, x), x).subs(x, -1)
                       + sp.diff(PJ(J, x), x).subs(x, 1)) == 0, J
print("P_J(-1)=1 and P'_J(-1)=-P'_J(1) for even J, D symbolic: OK")

# (c) large-J structure at fixed m: replace P_J-terms by their J-scaling.
#     The improvement term  -u^2/m^6 * 4u P'_J(1)/(m^4-u^2) at u=-p^2:
big = sp.simplify((-u**2 / m2**3) * 4 * u / (m2**2 - u**2))
print("coefficient of P'_J(1) in C2imp:", big, " at u=-p^2:",
      sp.simplify(big.subs(u, -p**2)), " (>0 for 0<p^2<m^2? ->",
      sp.simplify(big.subs(u, -p**2)) , ")")

# (d) J=4 smeared closed forms, D=7 and D=8, n=2 and n=3/2
for Dv, nn in [(7, 2), (8, sp.Rational(3, 2))]:
    ig = p**nn * C2imp(4, -p**2).subs(D, Dv)
    I = sp.simplify(sp.integrate(ig, (p, 0, 1)))
    print(f"D={Dv}, J=4, n={nn}: I(m2) =")
    print("   ", I)
