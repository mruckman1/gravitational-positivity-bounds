import sys, os, json
REPO="/Users/mruckman1/Desktop/dev/quantum_gravity"; sys.path.insert(0,REPO); os.chdir(REPO)
import sympy as sp, mpmath as mp
from qgse.verifiers.gravity_susy import SusyR4Verifier, D
from qgse.verifiers.gravity_positivity import _M2
from qgse.verifiers.gravity_lp import _W
d=json.load(open("results/positivity/artifacts/extu_tailhard_v2_functional.json")); c=d["config"]
a=[sp.Rational(s) for s in d["a_rat"]]
V=SusyR4Verifier(**{k:tuple(v) for k,v in c.items() if k.endswith("powers")}, p_max=sp.Rational(c["p_max"]))
w=_W; J=48
cols=V._columns(J); integ=sum(a[i]*cols[i] for i in range(len(cols)))
Eq,r0,r1,const=V._exact_smear_integral(integ,D,P=V.p_max); sub={_M2:1/w**2}
R0=sp.cancel(sp.together(Eq.subs(sub)/const.subs(sub))) if Eq!=0 else sp.Integer(0)
R1=sp.cancel(sp.together((r0*w).subs(sub)/const.subs(sub))) if r0!=0 else sp.Integer(0)
R2=sp.cancel(sp.together(r1.subs(sub)/(2*const.subs(sub)))) if r1!=0 else sp.Integer(0)
mp.mp.dps=200; Pf=mp.mpf(V.p_max.p)/V.p_max.q
pts={}
for w0 in [sp.Rational(q,100000) for q in range(90800,91800,100)]:
    r0v=sp.Rational(R0.subs(w,w0)) if R0!=0 else sp.Integer(0)
    r1v=sp.Rational(R1.subs(w,w0)) if R1!=0 else sp.Integer(0)
    r2v=sp.Rational(R2.subs(w,w0)) if R2!=0 else sp.Integer(0)
    x=mp.mpf(w0.p)/w0.q
    E=(mp.mpf(r0v.p)/r0v.q+(mp.mpf(r1v.p)/r1v.q)*mp.atan(Pf*x)+(mp.mpf(r2v.p)/r2v.q)*mp.log(1+(Pf*x)**2))
    pts[str(w0)]=float(E)
    print("w=%s m2=%.4f E=%.6e"%(w0,float(1/w0**2),float(E)),flush=True)
mn=min(pts.items(), key=lambda t:t[1])
json.dump({"functional":"extu_tailhard_v2 (c=3.3036, round-2)","J":J,
           "method":"exact rational R0/R1/R2 at rational w; atan/log at 200 dps",
           "points":pts,"min":{"w":mn[0],"E":mn[1]},
           "claim":"corner-window spike: grid scans read +0.58 while exact density is deeply negative"},
          open("results/positivity/artifacts/extu_tailhard_v2_j48_exact.json","w"),indent=1)
print("artifact written; min E = %.4e at w=%s"%(mn[1],mn[0]),flush=True)
