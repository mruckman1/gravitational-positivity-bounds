"""Load the persisted champion functional (no re-solve) and measure min_w E_J(w)
into the tail. Writes the tail to a permanent artifact."""
import sys, os, json
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); os.chdir(REPO)
import sympy as sp, mpmath as mp
mp.mp.dps = 40
from qgse.verifiers.gravity_susy import SusyR4Verifier, D
from qgse.verifiers.gravity_positivity import _M2
from qgse.verifiers.gravity_lp import _W

ART = os.path.join(REPO, "results/positivity/artifacts/extu_champion_j40_functional.json")
d = json.load(open(ART)); c = d["config"]
a_rat = [sp.Rational(s) for s in d["a_rat"]]
V = SusyR4Verifier(f_powers=tuple(c["f_powers"]), h_powers=tuple(c["h_powers"]),
                   e_powers=tuple(c["e_powers"]), x4_powers=tuple(c["x4_powers"]),
                   x6_powers=tuple(c["x6_powers"]), k_powers=tuple(c["k_powers"]),
                   e1_powers=tuple(c["e1_powers"]), p_max=sp.Rational(c["p_max"]))
Pq = V.p_max; nf_ = len(V.f_powers)
cG = sum(q * Pq**(n - 1) / (n - 1) for q, n in zip(a_rat[:nf_], V.f_powers))
print("persisted champion cG = %.9f   (expect 2.965141247)" % float(cG), flush=True)
print("audit sanity (must all PASS):", flush=True)
for J in (0, 20, 38, 40):
    ok, why = V.audit(a_rat, J, 40)
    print("  audit J=%2d: %s" % (J, "PASS" if ok else "FAIL " + str(why)), flush=True)

Pf = mp.mpf(Pq.p) / Pq.q; w = _W


def minE(J):
    cols = V._columns(J); integ = sum(a_rat[i] * cols[i] for i in range(len(cols)))
    Eq, r0, r1, const = V._exact_smear_integral(integ, D, P=Pq); sub = {_M2: 1 / w**2}
    R0 = sp.cancel(sp.together(Eq.subs(sub) / const.subs(sub))) if Eq != 0 else sp.Integer(0)
    R1 = sp.cancel(sp.together((r0 * w).subs(sub) / const.subs(sub))) if r0 != 0 else sp.Integer(0)
    R2 = sp.cancel(sp.together(r1.subs(sub) / (2 * const.subs(sub)))) if r1 != 0 else sp.Integer(0)
    f0 = sp.lambdify(w, R0, "mpmath"); f1 = sp.lambdify(w, R1, "mpmath"); f2 = sp.lambdify(w, R2, "mpmath")

    def E(x):
        x = mp.mpf(x)
        return f0(x) + f1(x) * mp.atan(Pf * x) + f2(x) * mp.log(1 + (Pf * x)**2)
    N = 2500; g = [mp.mpf(1) / N * i for i in range(1, N + 1)]
    e, xm = min(((E(x), x) for x in g), key=lambda t: t[0])
    for _ in range(28):
        st = mp.mpf(1) / N / 2
        for cc in (xm - st, xm + st):
            if 0 < cc <= 1:
                ev = E(cc)
                if ev < e:
                    e, xm = ev, cc
        N *= 2
    return e, xm


print("\nTRUE champion (g0<=2.96514) min_w E_J(w) into the tail:", flush=True)
out = {}
for J in range(40, 57, 2):
    e, xm = minE(J); out[J] = float(e)
    if abs(e) > mp.mpf('1e6'):
        tag = " <-- NUMERICAL (P_J lambdify unstable; disregard)"
    elif e < mp.mpf('-1e-9'):
        tag = " NEGATIVE"
    else:
        tag = " (ok, >=0)"
    print("  J=%3d: min_w E = %11s at w=%s%s" % (J, mp.nstr(e, 4), mp.nstr(xm, 4), tag), flush=True)
json.dump({"cG": float(cG), "tail_min_wEJ": out},
          open(os.path.join(REPO, "results/positivity/artifacts/extu_champion_j40_tail.json"), "w"), indent=1)
print("\nwrote results/positivity/artifacts/extu_champion_j40_tail.json", flush=True)
