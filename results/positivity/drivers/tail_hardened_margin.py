"""ROUND 4: the CLOSURE-READY functional. Same tail-hardened apparatus as
round 3 (extended-u champion basis, high-J rows J=42..176, audit in-loop
J<=60), plus the (E1)-informed margins the f_hat measurement showed are the
only remaining obstruction:
  - Bessel rows imposed with STRICT relative margin  row.a >= DELTA*|row|_1
    (round-3 functional dips to -3.4e-6 BETWEEN grid b's; f_hat scale at
    large b is itself ~1e-5-1e-7, so margins must be relative);
  - b-grid densified near the measured dips (b in [6,24] step 0.2) and
    extended to 320;
  - asymptotic sign row with strict margin (16 a_2 >= EPS, restoring manifest
    large-b dominance; round-3 sat exactly on a_2=0).
Shaping-only surgery: solve() is overridden (copied from the validated parent
with the margin b_ub change); the exact audit is untouched and remains the
sole arbiter."""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
import importlib.util
import numpy as np
import sympy as sp

import qgse.verifiers.gravity_susy as gs
from qgse.verifiers.gravity_susy import SusyR4Verifier, certify_g0_upper
from qgse.verifiers.gravity_lp import _linprog_retry, _bessel_transform_fn
from scipy.optimize import linprog as _lp

gs.linprog = lambda *a, **k: _linprog_retry(*a, tries=6, **k)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_rows import fast_row, validate_fast

ART = os.path.join(REPO, "results/positivity/artifacts/extu_margin_functional.json")
EXTRA_JS = (list(range(42, 82, 2)) + list(range(84, 130, 4))
            + list(range(136, 178, 8)))
DELTA = 1e-3       # relative Bessel-row margin: f_hat(b) >= DELTA * ||row||_1
EPS_A2 = 1e-4      # strict asymptotic margin: a_2 >= EPS_A2


def _m2grid(J):
    hi = max(120.0, (J / 2.0) ** 2)
    dip = (2.0 * J / 14.0) ** 2
    g = np.concatenate([np.geomspace(1.0, hi, 56),
                        np.linspace(0.5 * dip, 2.0 * dip, 20),
                        np.linspace(1.0, 1.35, 16),
                        np.linspace(1.38, 2.2, 8)])
    return np.unique(g[g >= 1.0])


class MarginVerifier(SusyR4Verifier):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.viability_rows = True
        self._extra_block = None

    # -- extra high-J rows (identical to round 3) ---------------------------
    def _build_extra(self):
        if self._extra_block is not None:
            return self._extra_block
        validate_fast(self, log=lambda m: print(m, flush=True))
        pm = float(self.p_max)
        rows, tags = [], []
        t0 = time.time()
        for J in EXTRA_JS:
            m2s = _m2grid(J)
            pg = np.linspace(1e-4, pm, max(240, int(2.2 * J)))
            for m2 in m2s:
                r = fast_row(self, J, float(m2), pg)
                rows.append(r / (np.abs(r).max() + 1e-300))
                tags.append((J, float(m2)))
        self._extra_block = (np.array(rows), tags)
        print("  [extra rows] BUILT %d high-J rows in %.0fs"
              % (len(rows), time.time() - t0), flush=True)
        return self._extra_block

    def rows_grid(self, j_max, n_xgrid, b_grid, extra_pts=()):
        base_rows, base_tags = super().rows_grid(j_max, n_xgrid, b_grid, extra_pts)
        # densify the Bessel grid near the measured f_hat dips + extend to 320
        tf = [_bessel_transform_fn(n, 10, float(self.p_max))
              for n in self.f_powers]
        nz = (len(self.h_powers) + len(self.e_powers) + len(self.x4_powers)
              + len(self.x6_powers) + len(self.x8_powers) + len(self.x10_powers)
              + len(self.k_powers) + len(self.e1_powers)
              + len(getattr(self, "e2_powers", ())))
        extra_bs = np.concatenate([np.arange(6.0, 24.01, 0.2),
                                   np.arange(161.0, 320.01, 1.0)])
        brows, btags = [], []
        for b in extra_bs:
            brows.append([t(float(b)) for t in tf] + [0.0] * nz)
            btags.append(("bessel", float(b)))
        xrows, xtags = self._build_extra()
        return (np.vstack([base_rows, np.array(brows), xrows]),
                base_tags + btags + xtags)

    # -- solve: copied from the validated parent, ONE change: Bessel rows get
    #    a strict margin b_ub = -DELTA*||row||_1 (and the asym row's margin is
    #    EPS_A2 in its own units) instead of plain feasibility 0. -------------
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
        A_eq = g0.reshape(1, -1)
        b_eq = np.array([sgn])
        if self.k_powers:
            nf_h = nf + nh
            off = nf_h + len(self.e_powers) + len(self.x4_powers) \
                + len(self.x6_powers)
            Pf_ = float(self.p_max)
            g2r = np.zeros(n); g3r = np.zeros(n)
            for j, i in enumerate(self.k_powers):
                g2r[off + j] = 2.0 * Pf_**(i + 1) / (i + 1)
                g3r[off + j] = Pf_**(i + 3) / (i + 3)
            A_eq = np.vstack([A_eq, g2r, g3r])
            b_eq = np.append(b_eq, [0.0, 0.0])
        # >>> the one change: strict margins on Bessel/asym rows <<<
        margins = np.zeros(len(rows))
        for i, t in enumerate(tags):
            if t[0] == "bessel":
                if t[1] == "asym_a2":
                    margins[i] = 16.0 * EPS_A2      # 16 a_2 >= 16*EPS_A2
                else:
                    margins[i] = DELTA * np.abs(rows[i]).sum()
        b_ub1 = -margins
        res = gs.linprog(gG, A_ub=-rows, b_ub=b_ub1,
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


def _main():
    def _load(p, n):
        s = importlib.util.spec_from_file_location(n, p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    
    
    ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_extu")
    champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "champ_m")
    cfg = champ.run_experiment()
    (fp, hp, epw, xp, x6p, kp, e1p, pm, cap, give, mr, clipped, ja) = ev._parse(cfg)
    print("ROUND 4 (closure-ready margins): DELTA=%g on Bessel rows, a_2>=%g; "
          "dense b in [6,24]+[161,320]" % (DELTA, EPS_A2), flush=True)
    V = MarginVerifier(f_powers=fp, h_powers=hp, e_powers=epw, x4_powers=xp,
                       x6_powers=x6p, k_powers=kp, e1_powers=e1p, p_max=pm)
    t0 = time.time()
    try:
        r = certify_g0_upper(j_max=40, j_audit=60, max_refine=10,
                             log=lambda m: print("  " + str(m), flush=True),
                             verifier=V, cap=cap, give=give, save_functional=ART)
    except Exception as e:
        print("RESULT: NO CERTIFICATE -- %s" % e, flush=True)
        sys.exit(2)
    print("\n=== CLOSURE-READY (margin) CERTIFICATE ===", flush=True)
    print("c_cert  = %.9f   (round-3 was 3.114449074)" % r["c_cert"], flush=True)
    print("exact   = %s..." % r["c_cert_exact"][:60], flush=True)
    print("iters   = %s   wall = %.0fs" % (r["iterations"], time.time() - t0), flush=True)


if __name__ == "__main__":
    _main()
