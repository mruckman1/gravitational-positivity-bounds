import sympy as sp

d = sp.Symbol('d', positive=True)
s, t, eps = sp.symbols('s t eps')
m2 = sp.Symbol('m2', positive=True)
Jc = sp.Symbol('X', nonnegative=True)
NORD = 5

def PJ_series_t(tvar, Rmax):
    out = 0
    for r in range(Rmax+1):
        num = sp.prod([Jc - i*(i+d-3) for i in range(r)]) if r>0 else sp.Integer(1)
        out += (2*tvar/m2)**r/(2**r*sp.factorial(r))*num/sp.rf((d-2)/2, r)
    return out

# CONSISTENT kernel: F = m^2 * G  (G = integrand of eq (16) rewritten)
PJ = PJ_series_t(t*eps, NORD)
F = m2*(2*m2+eps*t)/((m2-eps*s)*(m2+eps*s+eps*t)) * PJ/(m2*(m2+eps*t))
Fser = sp.expand(sp.series(F, eps, 0, NORD+1).removeO())
f = {}
for a in range(NORD+1):
    for b in range(NORD+1-a):
        f[(a,b)] = sp.simplify(Fser.coeff(eps,a+b).coeff(s,a).coeff(t,b))

# exact matches to eq (24):
checks = {
 "g2=<1/m^4>":            sp.simplify(f[(0,0)]/2 - 1/m2**2)==0,
 "g3=<(3-4X/(d-2))/m^6>":  sp.simplify(-f[(0,1)] - (3-4*Jc/(d-2))/m2**3)==0,
 "g4=<1/(2m^8)>":          sp.simplify(f[(2,0)]/4 - 1/(2*m2**4))==0,
 "g4=<(1+(4-5d)X/(2d(d-2))+X^2/(d(d-2)))/(2m^8)>":
     sp.simplify(f[(0,2)]/8 - (1+(4-5*d)/(2*d*(d-2))*Jc + Jc**2/(d*(d-2)))/(2*m2**4))==0,
}
for k,v in checks.items(): print(k, ":", v)

# null constraints
n4 = sp.simplify(2*f[(2,0)] - f[(0,2)])
r4 = sp.simplify(n4 / (((4-5*d)*Jc + 2*Jc**2)/m2**4))
print("n4/paper =", r4, " (const in m2, X:", r4.free_symbols, ")")
n5 = sp.simplify(f[(2,1)] - f[(0,3)])
r5 = sp.simplify(n5 / (((23*d**2-12*d-20)*Jc + (-21*d-2)*Jc**2 + 4*Jc**3)/m2**5))
print("n5/paper =", r5)

# d=4 explicit
print("d=4: n4 propto", sp.factor(sp.simplify((n4*m2**4).subs(d,4))))
print("d=4: n5 propto", sp.factor(sp.simplify((n5*m2**5).subs(d,4))))
print("d=4: g3 integrand:", sp.simplify((-f[(0,1)]).subs(d,4)))

# eq (53) check: v(x,J) rows
x = sp.Symbol('x', nonnegative=True)
M2 = sp.Symbol('M2', positive=True)
sub = {m2: M2*(1+x)}
pref = M2**2*(1+x)**4
row_g2 = sp.simplify((pref* (f[(0,0)]/2)).subs(d,4).subs(sub) * M2**2)   # M^4(1+x)^4 * g2fn ; g2fn=1/m^4
row_g3 = sp.simplify((pref* M2*(-f[(0,1)])).subs(d,4).subs(sub) * M2**2)
row_n4 = sp.simplify((pref* M2**2*(-16*Jc+2*Jc**2)/m2**4).subs(sub) * M2**2)
print("rows:", sp.expand(row_g2/M2**4), "|", sp.factor(row_g3/M2**4), "|", sp.simplify(row_n4/M2**4))
