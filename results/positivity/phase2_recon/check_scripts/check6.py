import sympy as sp
v = sp.Symbol('v', positive=True)  # mu^2/M^2 of the J=4 state
# primal: states (J=2,m2=1),(J=4,m2=v), weights a,b>0 with <n4>=0 -> a=20 b/v^4 (M=1,b=1)
a = 20/v**4
g2 = a + 1/v**2
g3 = -9*a - 37/v**3
f = -g3/g2  # = -g3tilde
vstar = sp.solve(sp.diff(sp.simplify(f),v), v)
vstar = [vv for vv in vstar if vv.is_real and vv>1]
print("v* =", vstar, "=", [float(x) for x in vstar])
fstar = sp.radsimp(sp.simplify(f.subs(v, vstar[0])))
alstar = sp.Rational(9,2)+7*sp.sqrt(305)/20
print("primal optimum =", sp.simplify(fstar), "; equals dual alpha*? ",
      sp.simplify(fstar - alstar)==0)
# NOTE: my weights: <F> = a F(1,J=2) + b F(v,J=4); g2=a*1/1^2+b/v^2 -> careful powers:
# g2fn=1/m^4 -> a/1 + 1/v^2, g3fn=(3-2X)/m^6 -> -9a - 37/v^3, n4: -24a + 480/v^4 =0 -> a=20/v^4. OK

# odd-spin trap: optimal even-spin functional is negative at J=3 (X=12)?
x = sp.Symbol('x', nonnegative=True)
bestar = (alstar-9)/24
G3 = alstar*(1+x)**2 + (1+x)*(3-2*12) + 2*bestar*12*(12-8)
xm = sp.nsolve(sp.diff(G3,x), 0.02)
print("J=3 block min:", float(G3.subs(x,xm)), "at x =", float(xm), "(negative => odd spins would change the optimum)")
# and J=2 block value at x=0 (should be exactly 0):
print("J=2 at x=0:", sp.simplify(alstar*1 + (3-12) - 48*bestar*... if False else alstar - 9 - 24*bestar))
