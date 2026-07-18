"""TAIL-HARDENED extended-u certificate (Stage-2 build, part 1).

Adds high-spin positivity to the LP shaping so the solved functional is positive
far past the old J<=40 audit scope:
  (a) direct per-state rows at J = 42..80 (step 2) and 84..120 (step 4),
      evaluated with the STABLE large-J machinery (_stable_row + hyp2f1 P_J --
      the expanded-polynomial lambdify is numerical garbage past J~60);
  (b) the extended Bessel b-grid to b=160 + the asymptotic large-b sign row
      (viability_rows=True);
then runs the UNCHANGED canonical certify (rationalize -> exact g2=g3=0
projection -> exact J<=40 audit) and PERSISTS the functional.

Shaping is the unsound layer -- nothing here touches the audit. The certificate
meaning: exact for the audited spins; the tail-hardening determines whether the
*same* functional stays positive beyond, which part 2 (extended audit J=42..60 +
stable tail scan to J~200) measures/proves per spin.
"""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
import importlib.util
import numpy as np
import mpmath as mp
import sympy as sp

import qgse.verifiers.gravity_susy as gs
from qgse.verifiers.gravity_susy import SusyR4Verifier, certify_g0_upper, D
from qgse.verifiers.gravity_lp import _linprog_retry

gs.linprog = lambda *a, **k: _linprog_retry(*a, tries=6, **k)

ART2 = os.path.join(REPO, "results/positivity/artifacts/extu_tailhard_functional.json")

EXTRA_JS = (list(range(42, 82, 2)) + list(range(84, 130, 4))
            + list(range(136, 178, 8)))


def _m2grid(J):
    """J-adaptive: the eikonal dip sits at b = 2J/m ~ O(10), i.e. m^2 ~ (J/7)^2
    -- it MOVES with J^2 (round-1 hole: J=68-70 negative at m2~92-100, exactly
    past the fixed m2<=81 grid). Cover b >= 4 (m2 up to (J/2)^2) with log
    spacing + a linear refill around the measured dip track b in [10, 20]."""
    hi = max(120.0, (J / 2.0) ** 2)
    dip = (2.0 * J / 14.0) ** 2
    g = np.concatenate([np.geomspace(1.0, hi, 56),
                        np.linspace(0.5 * dip, 2.0 * dip, 20),
                        # corner window m2 in [1, p_max^2~1.266]: for p_max>1
                        # the p=m kernel corner sits INSIDE the smear; measured
                        # round-2 violation: J=48, m2~1.203, E=-1.8e7 (narrow
                        # spike, invisible to coarse grids). Dense coverage.
                        np.linspace(1.0, 1.35, 16),
                        np.linspace(1.38, 2.2, 8)])
    return np.unique(g[g >= 1.0])


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_rows import fast_row, validate_fast


class TailHardenedVerifier(SusyR4Verifier):
    """Adds cached stable high-J rows on top of the standard grid; the extra
    block depends only on the (immutable) config, so it is built once and
    reused across refine iterations."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.viability_rows = True          # b->160 grid + asym a2 sign row
        self._extra_block = None            # (rows, tags) memo

    def _build_extra(self):
        if self._extra_block is not None:
            return self._extra_block
        # GATE: fast float64 rows must reproduce the production mpmath
        # _stable_row entrywise before any row is used (jet-heavy and
        # oscillation-heavy samples included).
        validate_fast(self, log=lambda m: print(m, flush=True))
        pm = float(self.p_max)
        rows, tags = [], []
        t0 = time.time()
        for J in EXTRA_JS:
            m2s = _m2grid(J)
            pg = np.linspace(1e-4, pm, max(240, int(2.2 * J)))
            for m2 in m2s:
                r = fast_row(self, J, float(m2), pg)
                # positivity rows are scale-invariant: normalize each row to
                # unit max-|entry| so J~120 rows (P_J ~ 1e50 near threshold)
                # do not wreck the LP conditioning.
                r = r / (np.abs(r).max() + 1e-300)
                rows.append(r)
                tags.append((J, float(m2)))
        self._extra_block = (np.array(rows), tags)
        print("  [extra rows] BUILT %d high-J rows in %.0fs (fast path)"
              % (len(rows), time.time() - t0), flush=True)
        return self._extra_block

    def rows_grid(self, j_max, n_xgrid, b_grid, extra_pts=()):
        base_rows, base_tags = super().rows_grid(j_max, n_xgrid, b_grid, extra_pts)
        xrows, xtags = self._build_extra()
        return np.vstack([base_rows, xrows]), base_tags + xtags


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_extu")
champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "champ_best")
cfg = champ.run_experiment()
(fp, hp, epw, xp, x6p, kp, e1p, pm, cap, give, mr, clipped, ja) = ev._parse(cfg)
print("champion basis, p_max=%s; tail-hardened shaping: J in %s..%s (%d spins), "
      "b-grid to 160 + asym row" % (pm, EXTRA_JS[0], EXTRA_JS[-1], len(EXTRA_JS)),
      flush=True)
V = TailHardenedVerifier(f_powers=fp, h_powers=hp, e_powers=epw, x4_powers=xp,
                         x6_powers=x6p, k_powers=kp, e1_powers=e1p, p_max=pm)
t0 = time.time()
try:
    # j_audit=60: the exact audit runs INSIDE the refinement loop for all
    # J<=60, so corner-window failures (round-2: J=48, m2~1.203) feed back as
    # refinement points exactly like the J<=40 ones did. Canonical scope of
    # the resulting certificate: J<=60 in one shot.
    r = certify_g0_upper(j_max=40, j_audit=60, max_refine=10,
                         log=lambda m: print("  " + str(m), flush=True),
                         verifier=V, cap=cap, give=give, save_functional=ART2)
except Exception as e:
    print("RESULT: NO CERTIFICATE -- %s" % e, flush=True)
    print("(If 'LP infeasible': the basis cannot support positivity through "
          "J<=120 -- a basis-limit finding. If HiGHS Status 15: gremlin, rerun.)",
          flush=True)
    sys.exit(2)
print("\n=== TAIL-HARDENED CERTIFICATE ===", flush=True)
print("c_cert  = %.9f   (truncated champion was 2.965141247; CMRS 3.000; "
      "string 2.4041)" % r["c_cert"], flush=True)
print("exact   = %s..." % r["c_cert_exact"][:60], flush=True)
print("scope   = %s" % r["scope"], flush=True)
print("iters   = %s   wall = %.0fs" % (r["iterations"], time.time() - t0), flush=True)
print("functional persisted -> %s" % ART2, flush=True)
