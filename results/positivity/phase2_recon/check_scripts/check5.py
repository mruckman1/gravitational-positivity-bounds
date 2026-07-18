import sympy as sp
x = sp.Symbol('x')

def nonneg_on_halfline(p):  # exact, rational coefficients
    p = sp.Poly(sp.expand(p), x, domain='QQ')
    if p.degree()==0: return p.LC() >= 0
    if p.eval(0) < 0 or p.LC() < 0: return False
    for fac, mult in p.factor_list()[1]:
        if mult % 2 == 1 and sp.Poly(fac,x).count_roots(sp.Rational(1,10**12), 10**12) > 0:
            return False
    return True

# rational functional just above the optimum: alpha = 106125/10000, beta=(alpha-9)/24
al = sp.Rational(106125,10000); be = (al-9)/24
G = lambda XX: al*(1+x)**2 + (1+x)*(3-2*XX) + 2*be*XX*(XX-8)
bad = [Jv for Jv in range(0,802,2) if not nonneg_on_halfline(G(Jv*(Jv+1)))]
print("alpha=10.6125 (rational cert): negative blocks J<=800:", bad)

# tail: G as quadratic in X for symbolic x: leading form positivity (impact param)
X = sp.Symbol('X')
Gx = al*(1+x)**2 + (1+x)*(3-2*X) + 2*be*X*(X-8)
discX = sp.discriminant(sp.Poly(Gx, X).as_expr(), X)
print("disc_X(G) as poly in x:", sp.factor(discX), "-> max over x>=0:",
      sp.maximum(discX, x, sp.Interval(0,sp.oo)))
# if disc_X < 0 for all x>=0 and coeff of X^2 >0, G>0 for ALL real X at every x>=0 (covers all J beyond any Jmax)
print("coeff X^2:", 2*be, "> 0")

# just below optimum fails (J=4)
alm = sp.Rational(10611,1000); bem=(alm-9)/24
print("alpha=10.611: J=4 block nonneg?", nonneg_on_halfline(alm*(1+x)**2+(1+x)*(3-40)+2*bem*20*12))
