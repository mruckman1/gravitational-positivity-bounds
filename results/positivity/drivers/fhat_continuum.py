"""(E1) RECONNAISSANCE: continuum behaviour of the impact-parameter transform
f_hat(b) = sum_n a_n t_n(b) for the persisted TAIL-HARDENED extended-u
functional (f-part only: the C_-2 column dominates the m->infty closure).
Decisive for the all-spin closure: if f_hat(b) < 0 anywhere, the closure FAILS
for this functional; if it has clean margin and the large-b asymptote
16 a_2 / b^3 dominates with the right sign, the closure is live and the
remaining work is rigor (explicit remainder constants), not construction.
Measurement, not proof."""
import sys, os, json
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import mpmath as mp
import sympy as sp
from qgse.verifiers.gravity_lp import _bessel_transform_fn

mp.mp.dps = 30
D = 10
ARTF = sys.argv[1] if len(sys.argv) > 1 else "results/positivity/artifacts/extu_tailhard_functional.json"
d = json.load(open(ARTF))
cfg = d["config"]
fp = cfg["f_powers"]
a_f = [float(sp.Rational(s)) for s in d["a_rat"][:len(fp)]]
print("f-part coefficients (n: a_n):", flush=True)
for n, a in zip(fp, a_f):
    print("  n=%2d: %+.6e" % (n, a), flush=True)
a2 = a_f[fp.index(2)] if 2 in fp else 0.0
print("asymptotic coefficient 16*a_2 = %+.6e  (must be > 0 for large-b "
      "positivity)" % (16 * a2), flush=True)

P_true = float(sp.Rational(cfg["p_max"]))
tf = [_bessel_transform_fn(n, D, P_true) for n in fp]   # TRUE endpoint


def fhat(b):
    return float(sum(a * t(float(b)) for a, t in zip(a_f, tf)))


# dense grid: linear to 40 (LP grid region), log 40..400 (beyond LP's b=160)
bs = np.unique(np.concatenate([np.linspace(0.05, 40.0, 400),
                               np.geomspace(40.0, 400.0, 220)]))
vals = np.array([fhat(b) for b in bs])
mn_i = int(np.argmin(vals))
print("\nf_hat(b) on b in [0.05, 400] (%d points):" % len(bs), flush=True)
print("  global min = %+.6e at b = %.3f" % (vals[mn_i], bs[mn_i]), flush=True)
neg = bs[vals < 0]
if len(neg):
    print("  NEGATIVE at %d points; first few b: %s" % (len(neg), neg[:8]),
          flush=True)
else:
    print("  NON-NEGATIVE at every sampled b", flush=True)
# near-zeros (margin < 1% of local scale)
scale = np.maximum.accumulate(np.abs(vals))[::-1][::-1]
low = [(float(b), float(v)) for b, v in zip(bs, vals)
       if v < 0.01 * (abs(vals).max() * (b < 5) + 1e-3) and v >= 0]
# asymptotic dominance check: f_hat(b) * b^3 -> 16 a_2 ?
print("\nlarge-b: f_hat(b) * b^3 vs 16*a_2 = %+.4e" % (16 * a2), flush=True)
for b in (60, 100, 160, 240, 320, 400):
    print("  b=%4d: f_hat*b^3 = %+.6e" % (b, fhat(b) * b**3), flush=True)
out = {"b": [float(x) for x in bs], "fhat": [float(v) for v in vals],
       "min": float(vals[mn_i]), "argmin_b": float(bs[mn_i]),
       "asym_16a2": 16 * a2}
json.dump(out, open(ARTF.replace("_functional.json", "_fhat.json"), "w"))
print("\nwrote %s" % ARTF.replace("_functional.json", "_fhat.json"), flush=True)
