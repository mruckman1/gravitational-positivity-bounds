"""TAIL-HARDENED p_max=1 certificate — the MINIMAL-AXIOM number (no A2/
meromorphy; non-perturbative). Same apparatus as the extended-u round-3 build
(high-J stable rows J=42..176, J^2-scaled m2 grids, b-grid to 160 + asym row,
exact audit in-loop J<=60), applied to the arena-1b champion basis (certified
3.0136 at J<=40). Answers: (a) the minimal-axiom tail-hardened value; (b) does
A2 still buy anything after hardening (vs extended-u 3.1144)?

Note: at p_max=1 the corner window m^2<p_max^2 is empty (m^2>=1) — the
corner-dense rows are harmless."""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
import sympy as sp

import qgse.verifiers.gravity_susy as gs
from qgse.verifiers.gravity_susy import SusyR4Verifier, certify_g0_upper
from qgse.verifiers.gravity_lp import _linprog_retry

gs.linprog = lambda *a, **k: _linprog_retry(*a, tries=6, **k)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_rows import fast_row, validate_fast

ART = os.path.join(REPO, "results/positivity/artifacts/pm1_tailhard_functional.json")
EXTRA_JS = (list(range(42, 82, 2)) + list(range(84, 130, 4))
            + list(range(136, 178, 8)))


def _m2grid(J):
    hi = max(120.0, (J / 2.0) ** 2)
    dip = (2.0 * J / 14.0) ** 2
    g = np.concatenate([np.geomspace(1.0, hi, 56),
                        np.linspace(0.5 * dip, 2.0 * dip, 20),
                        np.linspace(1.0, 1.35, 16),
                        np.linspace(1.38, 2.2, 8)])
    return np.unique(g[g >= 1.0])


class TailHardenedVerifier(SusyR4Verifier):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.viability_rows = True
        self._extra_block = None

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
                r = r / (np.abs(r).max() + 1e-300)
                rows.append(r); tags.append((J, float(m2)))
        self._extra_block = (np.array(rows), tags)
        print("  [extra rows] BUILT %d high-J rows in %.0fs"
              % (len(rows), time.time() - t0), flush=True)
        return self._extra_block

    def rows_grid(self, j_max, n_xgrid, b_grid, extra_pts=()):
        base_rows, base_tags = super().rows_grid(j_max, n_xgrid, b_grid, extra_pts)
        xrows, xtags = self._build_extra()
        return np.vstack([base_rows, xrows]), base_tags + xtags


cfg = json.load(open(os.path.join(
    REPO, "results/positivity/artifacts/arena1b_champion_D10.json")))["config"]
print("MINIMAL-AXIOM (p_max=1) tail-hardened build; arena-1b champion basis "
      "(J<=40 value 3.0136)", flush=True)
V = TailHardenedVerifier(
    f_powers=tuple(cfg["f_powers"]), h_powers=tuple(cfg["h_powers"]),
    e_powers=tuple(cfg["e_powers"]), x4_powers=tuple(cfg["x4_powers"]),
    x6_powers=tuple(cfg["x6_powers"]), k_powers=tuple(cfg["k_powers"]),
    e1_powers=tuple(cfg["e1_powers"]), p_max=sp.Rational(cfg["p_max"]))
t0 = time.time()
# cap: the arena-1b champion's 19700 lets coefficients outrun the K=26 audit
# sandwich (small-w refusals at J=16-20 with no refinement feedback). Cap at
# the validated 3000 regime (the extended-u champion audits cleanly there).
cap_use = min(float(cfg["cap"]), 3000.0)
print("using cap=%g (config had %g)" % (cap_use, float(cfg["cap"])), flush=True)
try:
    r = certify_g0_upper(j_max=40, j_audit=60, max_refine=10,
                         log=lambda m: print("  " + str(m), flush=True),
                         verifier=V, cap=cap_use, give=cfg["give"],
                         save_functional=ART)
except Exception as e:
    print("RESULT: NO CERTIFICATE -- %s" % e, flush=True)
    sys.exit(2)
print("\n=== MINIMAL-AXIOM TAIL-HARDENED CERTIFICATE (p_max=1) ===", flush=True)
print("c_cert  = %.9f   (J<=40 value was 3.013573408; CMRS 3.000; "
      "extended-u tail-hardened 3.114449)" % r["c_cert"], flush=True)
print("exact   = %s..." % r["c_cert_exact"][:60], flush=True)
print("scope   = %s" % r["scope"], flush=True)
print("iters   = %s   wall = %.0fs" % (r["iterations"], time.time() - t0), flush=True)
