"""Re-solve the extended-u champion from its stored config via the CLEAN default
path (reproduces 2.96514 byte-identically when the LP completes) and PERSIST the
exact-rational functional to a permanent artifact. Exits 0 on success (or if the
artifact already exists), 3 on the HiGHS Status-15 gremlin so an outer loop can
re-spawn a fresh process (the gremlin is per-process nondeterministic)."""
import sys, os, importlib.util
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
ART = os.path.join(REPO, "results/positivity/artifacts/extu_champion_j40_functional.json")
if os.path.exists(ART):
    print("functional already persisted:", ART); sys.exit(0)

import qgse.verifiers.gravity_susy as gs
from qgse.verifiers.gravity_lp import _linprog_retry
# result-preserving retry (returns the identical optimum on a completed solve);
# rides out transient non-completion without changing the vertex.
gs.linprog = lambda *a, **k: _linprog_retry(*a, tries=4, **k)


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_extu")
champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "champ_best")
cfg = champ.run_experiment()
(fp, hp, epw, xp, x6p, kp, e1p, pm, cap, give, mr, clipped, ja) = ev._parse(cfg)
from qgse.verifiers.gravity_susy import SusyR4Verifier, certify_g0_upper
V = SusyR4Verifier(f_powers=fp, h_powers=hp, e_powers=epw, x4_powers=xp,
                   x6_powers=x6p, k_powers=kp, e1_powers=e1p, p_max=pm)
try:
    r = certify_g0_upper(j_max=40, j_audit=ja, max_refine=mr, log=lambda m: None,
                         verifier=V, cap=cap, give=give, save_functional=ART)
except Exception as e:
    print("solve failed (this process's gremlin):", e); sys.exit(3)
ok = abs(r["c_cert"] - 2.965141246567263) < 1e-9
print("CERTIFIED c=%.9f  reproduces_champion=%s  saved=%s" % (r["c_cert"], ok, ART))
sys.exit(0 if ok else 4)
