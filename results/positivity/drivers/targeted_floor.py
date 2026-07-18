"""ROUND 5: TARGETED-MARGIN floor attack (data-driven from floor_sweep).

The give-sweep measured the barrier exactly:
 - raw hardened optima 2.72-2.90 but certification stalls at ~2.93-3.07;
 - ALL refusals live at low J (0..8), near-threshold-to-m~1.9 (the audit needs
   concentrated margin there; the uniform Chebyshev reserve dilutes it over
   ~10^4 rows);
 - at p_max>1 the refinement extra_pts go through the lambdify row path whose
   nan-guard corrupts them inside the corner window -> unbounded LPs.

Fixes here:
 (1) extra_pts routed through the STABLE fast_row path (normalized), never
     lambdify -- kills the unbounded-LP failure;
 (2) a LOW-J HARD-MARGIN block: J in {0..14}, m2 in a dense [1, 8] grid
     covering the measured dip band (m ~ 1.0-1.9), each row demanded at
     E >= DELTA_LOWJ * ||row||_1 -- margin bought exactly where the audit
     refuses, priced by the LP instead of donated by the reserve;
 (3) audit K-escalation to 36 (sound; done in gravity_lp).
Certify at give=0.003 (near-raw), j_audit=60 in-loop.
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

DELTA_LOWJ = 3e-3      # hard margin on the audit-critical low-J block


def _lowj_m2s():
    return np.unique(np.concatenate([np.geomspace(1.0002, 8.0, 72),
                                     np.linspace(1.9, 4.2, 24)]))


class TargetedVerifier(MarginVerifier):
    def rows_grid(self, j_max, n_xgrid, b_grid, extra_pts=()):
        # base WITHOUT extra_pts (their lambdify rows corrupt in the corner
        # window for p_max>1); stable margin rows appended below instead.
        rows, tags = super().rows_grid(j_max, n_xgrid, b_grid, extra_pts=())
        pm = float(self.p_max)
        add_r, add_t = [], []
        # (2) low-J hard-margin block
        for J in range(0, 16, 2):
            pg = np.linspace(1e-4, pm, 300)
            for m2 in _lowj_m2s():
                r = fast_row(self, J, float(m2), pg)
                add_r.append(r / (np.abs(r).max() + 1e-300))
                add_t.append(("lowj", J, float(m2)))
        # (1) refinement points, stable path, margin treatment
        for (J, m2) in extra_pts:
            pg = np.linspace(1e-4, pm, max(300, int(2.2 * J)))
            r = fast_row(self, int(J), float(m2), pg)
            add_r.append(r / (np.abs(r).max() + 1e-300))
            add_t.append(("lowj", int(J), float(m2)))
        return np.vstack([rows, np.array(add_r)]), tags + add_t

    def solve(self, side, j_max=40, n_xgrid=300, b_grid=(0.25, 80.0, 240),
              extra_pts=(), cap=3000.0, give=0.10):
        rows, tags = self.rows_grid(j_max, n_xgrid, b_grid, extra_pts)
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
        margins = np.zeros(len(rows))
        for i, t in enumerate(tags):
            if t[0] == "bessel":
                margins[i] = (16.0 * thm.EPS_A2 if t[1] == "asym_a2"
                              else thm.DELTA * np.abs(rows[i]).sum())
            elif t[0] == "lowj":
                margins[i] = DELTA_LOWJ * np.abs(rows[i]).sum()
        res = gs.linprog(gG, A_ub=-rows, b_ub=-margins,
                         A_eq=A_eq, b_eq=b_eq, bounds=[(-cap, cap)] * n,
                         method="highs")
        if not res.success:
            raise RuntimeError(f"LP infeasible/failed: {res.message}")
        opt = float(res.fun)
        scale = np.abs(rows).sum(axis=1) + 1e-12
        nrows = rows / scale[:, None]
        nmargins = margins / scale
        sl = np.array([[0.0] if t[0] == "bessel" else [1.0] for t in tags])
        c2 = np.zeros(n + 1); c2[-1] = -1.0
        A_ub2 = np.vstack([np.hstack([-nrows, sl]), np.append(gG, 0.0)])
        b_ub2 = np.append(-nmargins, opt + give * abs(opt) + 1e-9)
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


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_tf")
champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "champ_tf")
cfg = champ.run_experiment()
(fp, hp, epw, xp, x6p, kp, e1p, _pm, cap, _give, mr, clipped, ja) = ev._parse(cfg)

for pm in (sp.Rational(17, 16), sp.Rational(1)):
    tag = "pm" + str(pm).replace("/", "o")
    art = "results/positivity/artifacts/floor2_%s_functional.json" % tag
    print("\n=== TARGETED certify p_max=%s (DELTA_LOWJ=%g, give=0.003, "
          "K<=36 audit) ===" % (pm, DELTA_LOWJ), flush=True)
    V = TargetedVerifier(f_powers=fp, h_powers=hp, e_powers=epw, x4_powers=xp,
                         x6_powers=x6p, k_powers=kp, e1_powers=e1p, p_max=pm)
    t0 = time.time()
    try:
        r = certify_g0_upper(j_max=40, j_audit=60, max_refine=6,
                             log=lambda m: print("  " + str(m), flush=True),
                             verifier=V, cap=cap, give=0.003,
                             save_functional=art)
        print(">>> p_max=%s CERTIFIED c = %.9f  (%.0fs)  %s"
              % (pm, r["c_cert"], time.time() - t0,
                 "*** SUB-3.000 ***" if r["c_cert"] < 3.0 else ""), flush=True)
    except Exception as e:
        print(">>> p_max=%s REFUSED/FAILED: %s (%.0fs)"
              % (pm, str(e)[:80], time.time() - t0), flush=True)
print("\ndone", flush=True)
