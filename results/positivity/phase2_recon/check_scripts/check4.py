import sympy as sp

x, al, be, X = sp.symbols('x alpha beta X', real=True)
G = al*(1+x)**2 + (1+x)*(3-2*X) + 2*be*X*(X-8)   # functional rows, d=4, x=m^2/M^2-1

# saturation: J=2 root at x=0 ; J=4 double root
e1 = G.subs({X:6, x:0})                       # alpha - 9 - 24 beta = 0
G4 = sp.expand(G.subs(X,20))
e2 = sp.discriminant(G4, x)
sols = sp.solve([e1,e2],[al,be])
alstar = sp.Rational(9,2)+sp.Rational(7,4)*sp.sqrt(sp.Rational(61,5))
print("solutions:", sols)
sol = [sv for sv in sols if sv[0].is_real and sv[0]>0 and sv[0]==sp.nsimplify(sv[0])]
match = [sp.simplify(sv[0]-alstar)==0 for sv in sols]
print("alpha* = 9/2+(7/4)sqrt(61/5) in solutions:", match, "value:", float(alstar))
bestar = sp.simplify((alstar-9)/24)
print("beta* =", bestar, "=", float(bestar), " (paper: 0.0671875... from SDPB)")

# positivity check: even J to 400, exact Sturm-based nonnegativity on x>=0
def nonneg_on_halfline(p):
    p = sp.Poly(sp.expand(p), x)
    if p.degree()==0: return p.coeffs()[0] >= 0
    if any(c < 0 for c in [p.eval(0), p.LC()]): return False
    # count real roots in (0, oo) with odd multiplicity => sign change
    sq = sp.factor_list(p.as_expr())
    for fac, mult in sq[1]:
        if mult % 2 == 1:
            r = sp.Poly(fac, x)
            if r.count_roots(0, sp.oo) - (1 if r.eval(0)==0 else 0) > 0:
                # roots strictly inside (0,inf) of odd multiplicity -> fails
                # boundary root x=0 is ok if p(0)>=0 (touches zero)
                if r.count_roots(sp.Rational(1,10**9), sp.oo) > 0:
                    return False
    return True

Gs = G.subs({al: alstar, be: bestar})
bad = []
for Jv in range(0, 402, 2):
    p = sp.nsimplify(sp.expand(Gs.subs(X, Jv*(Jv+1))))
    if not nonneg_on_halfline(p): bad.append(Jv)
print("negative blocks for J<=400 at alpha*:", bad)

# J=4 double root location
r4 = sp.solve(sp.Eq(sp.diff(Gs.subs(X,20),x),0),x)
print("J=4 double root at x =", [float(sp.re(rr)) for rr in r4 if abs(sp.im(rr))<1e-12])

# strict failure just below optimum
Gm = G.subs({al: alstar - sp.Rational(1,100), be: (alstar-sp.Rational(1,100)-9)/24})
p4 = Gm.subs(X,20)
xm = sp.nsolve(sp.diff(p4,x), 0.75)
print("alpha*-0.01: J=4 block min value =", float(p4.subs(x,xm)), "(should be < 0)")

# impact-parameter / Regge polynomial q(xi) = alpha - 2 xi + 2 beta xi^2
q = alstar - 2*sp.Symbol('xi') + 2*bestar*sp.Symbol('xi')**2
print("disc(q) =", float(sp.discriminant(q, sp.Symbol('xi'))), "(negative => q>0 for all xi)")

# upper bound g3<=3 functional: 3*g2row - g3row >= 0 trivially
expr = sp.expand(3*(1+x)**2 - (1+x)*(3-2*X))
print("3*g2row - g3row =", sp.factor(expr), " (= 3x(1+x)+2X(1+x) >= 0 for x,X>=0)")
