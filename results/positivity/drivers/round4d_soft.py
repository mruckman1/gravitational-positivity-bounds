"""ROUND 4d: the E1'-CLOSABLE candidate. Margin recipe (record 3.0401's
apparatus) on the champion basis at p_max=9/8, plus the SOFT closure
structure the f_hat measurements prescribed:
  - |f(P)| <= PHI  (two inequality rows): shrinks the endpoint-oscillation
    amplitude of f_hat (the source of the recurring 1e-8 dips out to b~5600);
  - a_2 >= ALPHA   (variable bound): raises the smooth 16*a_2/b^3 floor so the
    dominance horizon b* ~ (A_osc/16*a_2)^{2/3} comes in to O(100), where the
    rigorous rational-series grid is tractable;
  - dense TRUE-endpoint b-rows to 600 (margined), as in 4c.
Goal is RIGOR (an f_hat provably positive on the continuum) near the record,
not a lower number. Usage: round4d_soft.py ALPHA PHI TAG
"""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
import importlib.util
import numpy as np
import sympy as sp

ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
PHI = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-2
TAG = sys.argv[3] if len(sys.argv) > 3 else "A"
GIVE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_hardened_margin as thm
from tail_hardened_margin import MarginVerifier
from fast_rows import fast_row
import qgse.verifiers.gravity_susy as gs
from qgse.verifiers.gravity_susy import certify_g0_upper
from qgse.verifiers.gravity_lp import _bessel_transform_fn


class SoftEndpointVerifier(MarginVerifier):
    def rows_grid(self, j_max, n_xgrid, b_grid, extra_pts=()):
        rows, tags = super().rows_grid(j_max, n_xgrid, b_grid, extra_pts=())
        pm = float(self.p_max)
        tf = [_bessel_transform_fn(n, 10, pm) for n in self.f_powers]
        nz = (len(self.h_powers) + len(self.e_powers) + len(self.x4_powers)
              + len(self.x6_powers) + len(self.x8_powers)
              + len(self.x10_powers) + len(self.k_powers)
              + len(self.e1_powers) + len(getattr(self, "e2_powers", ())))
        add_r, add_t = [], []
        for b in np.arange(321.0, 600.01, 1.0):
            add_r.append([t(float(b)) for t in tf] + [0.0] * nz)
            add_t.append(("bessel", float(b)))
        for (J, m2) in extra_pts:
            pg = np.linspace(1e-4, pm, max(300, int(2.2 * J)))
            add_r.append(list(fast_row(self, int(J), float(m2), pg)))
            add_t.append((int(J), float(m2)))
        return np.vstack([rows, np.array(add_r)]), tags + add_t

    def solve(self, side, j_max=40, n_xgrid=300, b_grid=(0.25, 80.0, 240),
              extra_pts=(), cap=3000.0, give=0.10):
        rows, tags = self.rows_grid(j_max, n_xgrid, b_grid, extra_pts)
        scale = np.abs(rows).sum(axis=1) + 1e-300
        rows = rows / scale[:, None]
        nf, nh = len(self.f_powers), len(self.h_powers)
        ne = (len(self.e_powers) + len(self.x4_powers)
              + len(self.x6_powers) + len(self.k_powers)
              + len(self.e1_powers)
              + len(self.x8_powers) + len(self.x10_powers)
              + len(getattr(self, "e2_powers", ())))
        n = nf + nh + ne
        Pf = float(self.p_max)
        gG = np.array([Pf**(p - 1) / (p - 1) for p in self.f_powers]
                      + [0.0] * (nh + ne))
        g0 = np.array([0.0] * nf
                      + [Pf**(i + 1) / (i + 1) for i in self.h_powers]
                      + [0.0] * ne)
        sgn = -1.0 if side == "upper" else 1.0
        A_eq = g0.reshape(1, -1); b_eq = np.array([sgn])
        if self.k_powers:
            off = nf + nh + len(self.e_powers) + len(self.x4_powers) \
                + len(self.x6_powers)
            g2r = np.zeros(n); g3r = np.zeros(n)
            for j, i in enumerate(self.k_powers):
                g2r[off + j] = 2.0 * Pf**(i + 1) / (i + 1)
                g3r[off + j] = Pf**(i + 3) / (i + 3)
            A_eq = np.vstack([A_eq, g2r, g3r])
            b_eq = np.append(b_eq, [0.0, 0.0])
        # soft endpoint rows: f(P) <= PHI and -f(P) <= PHI
        fP = np.zeros(n)
        for j, nn in enumerate(self.f_powers):
            fP[j] = Pf**nn
        ep_rows = np.vstack([fP, -fP])
        margins = np.zeros(len(rows))
        for i, t in enumerate(tags):
            if t[0] == "bessel":
                margins[i] = thm.EPS_A2 if t[1] == "asym_a2" else thm.DELTA
        i2 = list(self.f_powers).index(2)
        bounds1 = [(-cap, cap)] * n
        bounds1[i2] = (ALPHA, cap)                     # a_2 >= ALPHA
        res = gs.linprog(gG, A_ub=np.vstack([-rows, ep_rows]),
                         b_ub=np.concatenate([-margins, [PHI, PHI]]),
                         A_eq=A_eq, b_eq=b_eq, bounds=bounds1,
                         method="highs")
        if not res.success:
            raise RuntimeError(f"LP infeasible/failed: {res.message}")
        opt = float(res.fun)
        sl = np.array([[0.0] if t[0] == "bessel" else [1.0] for t in tags])
        c2 = np.zeros(n + 1); c2[-1] = -1.0
        A_ub2 = np.vstack([np.hstack([-rows, sl]),
                           np.hstack([ep_rows, np.zeros((2, 1))]),
                           np.append(gG, 0.0)])
        b_ub2 = np.concatenate([-margins, [PHI, PHI],
                                [opt + give * abs(opt) + 1e-9]])
        A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
        for cap2 in (300.0, 1000.0, cap):
            bounds2 = [(-cap2, cap2)] * n + [(0, None)]
            bounds2[i2] = (ALPHA, cap2)
            r2 = gs.linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq,
                            bounds=bounds2, method="highs")
            if r2.success and r2.x[-1] > 1e-9:
                return r2.x[:n], float(gG @ r2.x[:n]), float(r2.x[-1])
        return res.x, opt, 0.0


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_4d")
champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "ch_4d")
cfg = champ.run_experiment()
(fp, hp, epw, xp, x6p, kp, e1p, pm, cap, _g, mr, clipped, ja) = ev._parse(cfg)
V = SoftEndpointVerifier(f_powers=fp, h_powers=hp, e_powers=epw, x4_powers=xp,
                         x6_powers=x6p, k_powers=kp, e1_powers=e1p, p_max=pm)
ART = "results/positivity/artifacts/extu_soft_%s_functional.json" % TAG
print("=== ROUND 4d[%s]: soft endpoint |f(P)|<=%g, a_2>=%g, give=%g (p_max=%s) ==="
      % (TAG, PHI, ALPHA, GIVE, pm), flush=True)
t0 = time.time()
try:
    r = certify_g0_upper(j_max=40, j_audit=60, max_refine=6,
                         log=lambda m: print("  " + str(m), flush=True),
                         verifier=V, cap=cap, give=GIVE, save_functional=ART)
    print(">>> CERTIFIED c = %.9f (%.0fs)" % (r["c_cert"], time.time() - t0),
          flush=True)
except Exception as e:
    print(">>> REFUSED/FAILED: %s (%.0fs)" % (str(e)[:100], time.time() - t0),
          flush=True)
print("done", flush=True)
