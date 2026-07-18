import sympy as sp

d = sp.Symbol('d', positive=True)
s, t, eps = sp.symbols('s t eps')
m2 = sp.Symbol('m2', positive=True)
Jc = sp.Symbol('X', nonnegative=True)   # Casimir J(J+d-3)
NORD = 5  # total degree in (s,t) kept

# P_J(1+2t/m^2) as series in t, symbolic Casimir, general d
def PJ_series_t(tvar, Rmax):
    out = 0
    for r in range(Rmax+1):
        num = sp.prod([Jc - i*(i+d-3) for i in range(r)]) if r>0 else sp.Integer(1)
        poch = sp.rf((d-2)/2, r)
        out += (2*tvar/m2)**r/(2**r*sp.factorial(r))*num/poch
    return out

PJ = PJ_series_t(t*eps, NORD)
K = (2*m2+eps*t)/((m2-eps*s)*(m2+eps*s+eps*t)) * PJ/(m2*(m2+eps*t))
Kser = sp.series(K, eps, 0, NORD+1).removeO()
Kser = sp.expand(Kser)

# f[a,b] = coefficient of s^a t^b (function of m2, X, d)
f = {}
for a in range(NORD+1):
    for b in range(NORD+1-a):
        c = Kser.coeff(eps, a+b).coeff(s, a).coeff(t, b)
        f[(a,b)] = sp.simplify(c)

# LHS coefficients from check1:
g2c,g3c,g4c,g5c,g6c,g6p = sp.symbols('g2c g3c g4c g5c g6c g6p')
LHS = {(0,0):2*g2c, (0,1):-g3c, (2,0):4*g4c, (1,1):4*g4c, (0,2):8*g4c,
       (3,0):sp.Integer(0), (2,1):-2*g5c, (1,2):-2*g5c, (0,3):-2*g5c,
       (4,0):8*g6c, (3,1):16*g6c, (2,2):32*g6c+g6p, (1,3):24*g6c+g6p, (0,4):24*g6c}

# ---- moments ----
print("g2 = <", sp.simplify(f[(0,0)]/2), ">")
print("g3 = <", sp.simplify(-f[(0,1)]), ">")
print("g4 (from s^2) = <", sp.simplify(f[(2,0)]/4), ">")
print("g4 (from t^2) = <", sp.simplify(f[(0,2)]/8), ">")

# check vs paper eq (24)
g2p = 1/m2**2
g3p = (3 - 4*Jc/(d-2))/m2**3
g4p_a = 1/(2*m2**4)
g4p_b = (1 + (4-5*d)/(2*d*(d-2))*Jc + Jc**2/(d*(d-2)))/(2*m2**4)
print("match g2:", sp.simplify(f[(0,0)]/2 - g2p)==0)
print("match g3:", sp.simplify(-f[(0,1)] - g3p)==0)
print("match g4a:", sp.simplify(f[(2,0)]/4 - g4p_a)==0)
print("match g4b:", sp.simplify(f[(0,2)]/8 - g4p_b)==0)

# ---- null constraints: n4 from 2*f(2,0)-f(0,2) etc ----
n4 = sp.simplify(2*f[(2,0)] - f[(0,2)])
n4paper = ((4-5*d)*Jc + 2*Jc**2)/m2**4
print("n4 =", sp.factor(n4), " ; ratio to paper:", sp.simplify(n4/n4paper))

# degree-5: LHS rows (2,1),(1,2),(0,3) all equal -2 g5. Two independent zero-LHS combos:
c1 = sp.simplify(f[(2,1)] - f[(1,2)])
c2 = sp.simplify(f[(2,1)] - f[(0,3)])
n5paper = ((23*d**2-12*d-20)*Jc + (-21*d-2)*Jc**2 + 4*Jc**3)/m2**5
print("c1 (pointwise?):", sp.simplify(c1))
print("c2 ratio to paper n5:", sp.simplify(c2/n5paper))
# also s-odd rows pointwise identities check: f(1,1) vs relation? f(3,0)?
print("f(3,0):", sp.simplify(f[(3,0)]), " (should be expressible via others; LHS=0)")
