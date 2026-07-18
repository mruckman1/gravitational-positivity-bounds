import sympy as sp

d = sp.Symbol('d', positive=True)
s, t = sp.symbols('s t')
m2 = sp.Symbol('m2', positive=True)
Jc = sp.Symbol('Jcas', nonnegative=True)  # Casimir J(J+d-3)

# ---- n_J^{(d)} at d=4 ----
Jn = sp.Symbol('Jn', integer=True, nonnegative=True)
nJd = (4*sp.pi)**(d/2)*(d+2*Jn-3)*sp.gamma(d+Jn-3)/(sp.pi*sp.gamma((d-2)/2)*sp.gamma(Jn+1))
print("n_J^(4)/(16 pi (2J+1)) =", sp.simplify(nJd.subs(d,4)/(16*sp.pi*(2*Jn+1))))

# ---- P_J(1+y) Taylor around 1 ----
def PJ_series(y, Rmax, dval):
    out = 0
    for r in range(Rmax+1):
        num = sp.prod([Jc - i*(i+dval-3) for i in range(r)]) if r>0 else 1
        poch = sp.rf(sp.Rational(dval-2,2) if isinstance(dval,int) else (dval-2)/2, r)
        out += y**r/(2**r*sp.factorial(r))*num/poch
    return out

yv = sp.Rational(3,7)
for Jv in [0,2,4,6]:
    exact = sp.legendre(Jv, 1+yv)
    ser = PJ_series(yv, Jv, 4).subs(Jc, Jv*(Jv+1))
    assert sp.simplify(exact-ser)==0, Jv
print("P_J(1+y) Taylor formula matches Legendre (d=4): OK")

# ---- LHS of eq (16): low-energy side ----
g2c,g3c,g4c,g5c,g6c,g6p = sp.symbols('g2c g3c g4c g5c g6c g6p')
sig2 = s**2+t**2+(s+t)**2
sig3 = s*t*(-s-t)
Mlow = g2c*sig2 + g3c*sig3 + g4c*sig2**2 + g5c*sig2*sig3 + g6c*sig2**3 + g6p*sig3**2
sp_ = sp.Symbol('sprime')
expr_res = Mlow.subs(s, sp_)/((sp_-s)*sp_*(sp_+t))
LHS = Mlow/(s*(s+t)) + sp.residue(expr_res, sp_, 0) + sp.residue(expr_res, sp_, -t)
LHS = sp.expand(sp.cancel(sp.together(LHS)))
print("LHS =", LHS)
