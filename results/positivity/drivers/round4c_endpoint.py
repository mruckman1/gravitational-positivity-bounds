"""ROUND 4c: the CLOSURE-STRUCTURED functional. Margin recipe (round 4) plus:
 - TRUE-endpoint Bessel rows everywhere (pmax=p_max, the found+fixed default);
 - dense true-endpoint b-rows extended to b=600;
 - TWO new exact linear constraints: f(P)=0 and f'(P)=0 (smearing profile
   vanishes at the endpoint to first order). This kills the leading endpoint
   oscillation of f_hat (amplitude ~ f(P)/b^{4.5}, then f'(P)/b^{5.5}) that
   produced recurring 1e-8 dips out to b~5000 past any finite grid; with both
   zero, the smooth a_3/b^4 term dominates and the E1' tail argument becomes a
   clean explicit-dominance proof.
give from round-4 (0.05 equivalent: use champion cfg give), j_audit=60 in-loop.
"""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
import importlib.util
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_hardened_margin as thm
from tail_hardened_margin import MarginVerifier
from fast_rows import fast_row
import qgse.verifiers.gravity_susy as gs
from qgse.verifiers.gravity_susy import certify_g0_upper
from qgse.verifiers.gravity_lp import _bessel_transform_fn


class EndpointVerifier(MarginVerifier):
    """Margin recipe + endpoint-vanishing profile (f(P)=f'(P)=0) + true-
    endpoint dense b rows to 600 + all-row L1 normalization."""

    def rows_grid(self, j_max, n_xgrid, b_grid, extra_pts=()):
        # base: standard grid + viability bessel (true endpoint now, via the
        # fixed gravity_susy source) + high-J block (MarginVerifier adds its
        # own dense b rows [6,24]+[161,320], now true-endpoint too)
        rows, tags = super().rows_grid(j_max, n_xgrid, b_grid, extra_pts=())
        pm = float(self.p_max)
        # extend true-endpoint b rows 321..600
        tf = [_bessel_transform_fn(n, 10, pm) for n in self.f_powers]
        nz = (len(self.h_powers) + len(self.e_powers) + len(self.x4_powers)
              + len(self.x6_powers) + len(self.x8_powers)
              + len(self.x10_powers) + len(self.k_powers)
              + len(self.e1_powers))
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
              + len(self.x8_powers) + len(self.x10_powers))
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
        # >>> endpoint-vanishing profile: f(P) = 0 and f'(P) = 0 <<<
        fP = np.zeros(n); fpP = np.zeros(n)
        for j, nn in enumerate(self.f_powers):
            fP[j] = Pf**nn
            fpP[j] = nn * Pf**(nn - 1)
        A_eq = np.vstack([A_eq, fP, fpP])
        b_eq = np.append(b_eq, [0.0, 0.0])
        margins = np.zeros(len(rows))
        for i, t in enumerate(tags):
            if t[0] == "bessel":
                margins[i] = thm.EPS_A2 if t[1] == "asym_a2" else thm.DELTA
        res = gs.linprog(gG, A_ub=-rows, b_ub=-margins,
                         A_eq=A_eq, b_eq=b_eq, bounds=[(-cap, cap)] * n,
                         method="highs")
        if not res.success:
            raise RuntimeError(f"LP infeasible/failed: {res.message}")
        opt = float(res.fun)
        sl = np.array([[0.0] if t[0] == "bessel" else [1.0] for t in tags])
        c2 = np.zeros(n + 1); c2[-1] = -1.0
        A_ub2 = np.vstack([np.hstack([-rows, sl]), np.append(gG, 0.0)])
        b_ub2 = np.append(-margins, opt + give * abs(opt) + 1e-9)
        A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
        for cap2 in (300.0, 1000.0, cap):
            r2 = gs.linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq,
                            bounds=[(-cap2, cap2)] * n + [(0, None)],
                            method="highs")
            if r2.success and r2.x[-1] > 1e-9:
                return r2.x[:n], float(gG @ r2.x[:n]), float(r2.x[-1])
        return res.x, opt, 0.0


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_4c")
champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "ch_4c")
cfg = champ.run_experiment()
(fp, hp, epw, xp, x6p, kp, e1p, pm, cap, give, mr, clipped, ja) = ev._parse(cfg)
V = EndpointVerifier(f_powers=fp, h_powers=hp, e_powers=epw, x4_powers=xp,
                     x6_powers=x6p, k_powers=kp, e1_powers=e1p, p_max=pm)
ART = "results/positivity/artifacts/extu_endpoint_functional.json"
print("=== ROUND 4c: endpoint-vanishing closure-structured functional "
      "(p_max=%s, f(P)=f'(P)=0, true-endpoint b-rows to 600) ===" % pm,
      flush=True)
t0 = time.time()
try:
    r = certify_g0_upper(j_max=40, j_audit=60, max_refine=6,
                         log=lambda m: print("  " + str(m), flush=True),
                         verifier=V, cap=cap, give=0.05, save_functional=ART)
    print(">>> CERTIFIED c = %.9f (%.0fs)" % (r["c_cert"], time.time() - t0),
          flush=True)
except Exception as e:
    print(">>> REFUSED/FAILED: %s (%.0fs)" % (str(e)[:100], time.time() - t0),
          flush=True)
print("done", flush=True)
