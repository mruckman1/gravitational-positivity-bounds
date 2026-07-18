import sympy as sp

s,t = sp.symbols('s t')
u = -s-t
A = sp.expand(s**2+t**2+u**2)
B = sp.expand(s*t*u)

# spin-0 extremal amplitude, m=1
M = 1/(1-s)+1/(1-t)+1/(1-u)
ser = sp.expand(sp.series(M.rewrite(sp.Pow), s, 0, 12).removeO())
# safer: full 2-var Taylor via geometric series
M2 = sum(s**n + t**n + sp.expand(u**n) for n in range(0,12))
M2 = sp.expand(M2)

# express in terms of A, B: match monomials A^a B^b with 2a+3b = degree
aa, bb = sp.symbols('aa bb')
from itertools import product
poly = sp.Poly(M2, s, t)
# build basis
basis = {}
for a in range(0,7):
    for b in range(0,5):
        deg = 2*a+3*b
        if deg <= 11:
            basis[(a,b)] = sp.expand(A**a * B**b)

# solve linear system per degree
coeffs = {}
remaining = M2
for deg in range(0,12):
    part = sum(c*mon for mon, c in zip(poly.monoms(), poly.coeffs()) if False)
# simpler: collect homogeneous parts
def homog(expr, deg):
    e = sp.expand(expr)
    out = 0
    for term in e.as_ordered_terms():
        p = sp.Poly(term, s, t)
        if p.total_degree() == deg:
            out += term
    return out

for deg in range(2,12):
    target = homog(M2, deg)
    cands = [(a,b) for (a,b) in basis if 2*a+3*b == deg]
    unk = sp.symbols(f'c0:{len(cands)}')
    expr = sp.expand(target - sum(unk[i]*basis[c] for i,c in enumerate(cands)))
    sol = sp.solve([sp.Eq(cf,0) for cf in sp.Poly(expr, s, t).coeffs()], unk, dict=True)
    if sol:
        for i,c in enumerate(cands):
            coeffs[c] = sol[0].get(unk[i], 0)

g2 = coeffs[(1,0)]
print("g2 =", g2)
names = {(1,0):'g2 (A)', (0,1):'g3 (B)', (2,0):'g4 (A^2)', (1,1):'g5 (AB)', (3,0):'g6 (A^3)', (0,2):"g6' (B^2)",
         (2,1):'g7 (A^2 B)', (4,0):'g8 (A^4)', (1,2):"g8' (A B^2)", (3,1):'g9 (A^3 B)', (0,3):"g9' (B^3)",
         (5,0):'g10 (A^5)', (2,2):"g10' (A^2 B^2)"}
for key,name in names.items():
    if key in coeffs:
        print(f"{name}: gtilde = {sp.nsimplify(coeffs[key]/g2)} = {float(coeffs[key]/g2):.4f}")

# alpha_* closed form and kappa(D) checks
import mpmath as mp
d = 4
alpha = (5*d-2 + mp.sqrt((d+3)*(319*d**3+76*d**2-292*d+32)/(6*(d+1)*(d+4))))/(2*(d-2))
print("\nalpha_*(4) =", alpha)
print("9/2+7/4*sqrt(61/5) =", 4.5+1.75*mp.sqrt(61/5.))
for D in (4,6,10):
    kap = mp.sqrt((D+3)*(319*D**3+76*D**2-292*D+32)/(24*(D-2)**2*(D+1)*(D+4))) + 6*(5*D-2)/(12*(D-2))
    al  = (5*D-2 + mp.sqrt((D+3)*(319*D**3+76*D**2-292*D+32)/(6*(D+1)*(D+4))))/(2*(D-2))
    print(f"D={D}: kappa={kap}, alpha*={al}, diff={kap-al}")
# warm-up bound d=4
print("2(5d-4)/(d-2) at d=4 =", 2*(5*4-4)/(4-2))
# n_4 in d=4
J = sp.symbols('J')
Jc2 = J*(J+1)
n4 = sp.expand(Jc2*(2*Jc2-16))
print("n4 d=4:", sp.factor(n4), " = 2J(J+1)[J(J+1)-8]:", sp.expand(2*Jc2*(Jc2-8))==n4)
# J_critical between 2 and 3 for all d>=3?
for dd in (3,4,6,10,26):
    jc = (3-dd+mp.sqrt(dd*(dd+4)+1))/2
    print("d=",dd,"J_crit=",jc)
# string check: s*tan(pi*s) series
x = sp.symbols('x')
print("\ns*tan(pi s):", sp.series(x*sp.tan(sp.pi*x), x, 0, 8))
# Goldstone: lambda/Mh^2 * [-s^2/(s-Mh^2) - t^2/(t-Mh^2) - u^2/(u-Mh^2)] at t->0, Mh=1
lam = sp.symbols('lambda')
Afor = lam*(-s**2/(s-1) - 0 - sp.expand((-s)**2)/(-s-1))
print("Goldstone forward:", sp.series(sp.simplify(Afor), s, 0, 8))
