"""ROUND 9 (basis expansion probe): R7 recipe + the round-4 recipe (Bessel margins ONLY, generous reserve) at
p_max=17/16 -- the better raw optimum -- with all rows L1-normalized (the
conditioning fix that killed the bogus 'unbounded'). No low-J margin blocks:
round 6 measured those as a +0.2 provability tax; here the reserve donates
margin instead, as in the 3.0401 record. give=0.05, j_audit=60, K<=36."""
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


class R7Verifier(MarginVerifier):
    """MarginVerifier + L1 row normalization + stable extra_pts. No low-J
    margin blocks -- the reserve supplies audit margin (round-4 recipe)."""

    def rows_grid(self, j_max, n_xgrid, b_grid, extra_pts=()):
        rows, tags = super().rows_grid(j_max, n_xgrid, b_grid, extra_pts=())
        pm = float(self.p_max)
        add_r, add_t = [], []
        # ultra-fine threshold slice: the 9b blocker lived at m2~1.0006,
        # BETWEEN the corner grid's first points. Hard margin here.
        for J in range(0, 42, 2):
            pg = np.linspace(1e-4, pm, max(300, int(2.2 * J)))
            for m2 in np.linspace(1.0001, 1.015, 25):
                add_r.append(fast_row(self, J, float(m2), pg))
                add_t.append(("thr", J, float(m2)))
        for (J, m2) in extra_pts:
            pg = np.linspace(1e-4, pm, max(300, int(2.2 * J)))
            add_r.append(fast_row(self, int(J), float(m2), pg))
            add_t.append((int(J), float(m2)))       # ordinary (m,J) row: slack
        if add_r:
            rows = np.vstack([rows, np.array(add_r)])
            tags = tags + add_t
        return rows, tags

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
        margins = np.zeros(len(rows))
        for i, t in enumerate(tags):
            if t[0] == "bessel":
                margins[i] = thm.EPS_A2 if t[1] == "asym_a2" else thm.DELTA
            elif t[0] == "thr":
                margins[i] = 6e-3          # hard threshold-slice margin
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


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_r7")
champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "ch_r7")
cfg = champ.run_experiment()
(fp, hp, epw, xp, x6p, kp, e1p, _pm, cap, _g, mr, clipped, ja) = ev._parse(cfg)

pm = sp.Rational(9, 8)
V = R7Verifier(f_powers=fp, h_powers=hp, e_powers=epw, x4_powers=xp,
               x6_powers=x6p, k_powers=kp, e1_powers=e1p, p_max=pm,
               x8_powers=(0, 2, 4, 6), e2_powers=(0, 2, 4, 6, 8))
ART = "results/positivity/artifacts/round9c_basis_functional.json"
print("=== ROUND 9c (threshold slice + give=0.16): margin recipe at p_max=9/8 + X8(0,2,4,6) + E2(0,2,4,6,8) towers ===", flush=True)
t0 = time.time()
try:
    r = certify_g0_upper(j_max=40, j_audit=60, max_refine=6,
                         log=lambda m: print("  " + str(m), flush=True),
                         verifier=V, cap=cap, give=0.16, save_functional=ART)
    print(">>> CERTIFIED c = %.9f (%.0fs) %s"
          % (r["c_cert"], time.time() - t0,
             "*** BELOW 3.0401 RECORD ***" if r["c_cert"] < 3.0401 else ""),
          flush=True)
except Exception as e:
    print(">>> REFUSED/FAILED: %s (%.0fs)" % (str(e)[:100], time.time() - t0),
          flush=True)
print("done", flush=True)
