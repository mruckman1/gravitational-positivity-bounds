"""Direct probe of the k-tower (C2^imp-susy) fitness landscape — the mechanism
behind whether machine guidance matters. The greedy/GA arms rarely VISIT
k-populated designs (populating k hurts immediate fitness, so they reject it);
this probe evaluates the landscape directly, independent of whether the search
stumbles into it.

Designs (all via the IDENTICAL evaluator, forked-child + EWMA-style hard cap):
  champ        : the LLM champion (expect ~2.96514) — sanity anchor
  champ_no_k   : champion with k-tower REMOVED — is the tower essential?
  champ_no_e1  : champion with e1-tower removed
  seed         : gen-0 seed (k,e1 empty) — expect ~3.06
  seed+k       : seed + ONLY the champion k-tower added — does one structural
                 add help or hurt? (valley test)
  seed+e1      : seed + ONLY e1 added
  seed+k+e1    : seed + both towers (but seed's other columns) — partial assembly
  k2,k23,k234  : champion with k truncated — is the FULL tower needed?

Output: certified c for each (or refusal), so we can see whether the good value
only appears when the whole structure is assembled = a fitness valley local
search won't cross.
"""
import sys, os, json, time, copy
import multiprocessing as _mp
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
import importlib.util
def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_probe")
init = _load(os.path.join(REPO, "tasks/grav_extu/initial.py"), "init_probe")
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "kvalley_probe.jsonl")
CAP = 1200.0     # hard per-eval wall cap (like the EWMA tail)
_CTX = _mp.get_context("fork")

SEED = init.propose_design()
CHAMP = json.load(open("results/positivity/artifacts/extu_champion_j40.json"))
CHAMP = CHAMP.get("config", CHAMP)
# normalize champion to a design dict the evaluator accepts
CH = {k: CHAMP[k] for k in ("f_powers", "h_powers", "e_powers", "x4_powers",
      "x6_powers", "k_powers", "e1_powers", "p_max", "cap", "give", "j_audit",
      "max_refine") if k in CHAMP}
CH.setdefault("p_max", "9/8"); CH.setdefault("cap", 15000.0)
CH.setdefault("give", 0.05); CH.setdefault("j_audit", 40); CH.setdefault("max_refine", 5)

def d(base, **over):
    x = copy.deepcopy(base); x.update(over); return x

DESIGNS = {
    "champ":       CH,
    "champ_no_k":  d(CH, k_powers=[]),
    "champ_no_e1": d(CH, e1_powers=[]),
    "champ_k2":    d(CH, k_powers=[2, 3]),
    "champ_k234":  d(CH, k_powers=[2, 3, 4]),
    "seed":        SEED,
    "seed+k":      d(SEED, k_powers=[2, 3, 4, 5], j_audit=40),
    "seed+e1":     d(SEED, e1_powers=[0, 2, 4, 6, 8], j_audit=40),
    "seed+k+e1":   d(SEED, k_powers=[2, 3, 4, 5], e1_powers=[0, 2, 4, 6, 8], j_audit=40),
}

def _child(dd, q):
    try:
        m = ev.aggregate([dd])
        q.put(("ok", m.get("public", {})))
    except Exception as e:                       # noqa
        q.put(("err", {"status": "error", "err": str(e)[:200]}))

def run(name, dd):
    q = _CTX.Queue(); p = _CTX.Process(target=_child, args=(dd, q))
    t0 = time.time(); p.start(); p.join(CAP)
    if p.is_alive():
        p.terminate(); p.join(5); p.kill() if p.is_alive() else None
        pub = {"status": "timeout"}
    else:
        try: _, pub = q.get_nowait()
        except Exception: pub = {"status": "no_result"}
    wall = time.time() - t0
    rec = {"name": name, "status": pub.get("status"), "c": pub.get("c"),
           "k_len": len(dd.get("k_powers", [])), "e1_len": len(dd.get("e1_powers", [])),
           "j_audit": dd.get("j_audit"), "wall_s": round(wall, 1)}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("[probe] %-12s status=%-10s c=%-10s k=%d e1=%d (%.0fs)" %
          (name, str(pub.get("status")), str(pub.get("c")), rec["k_len"],
           rec["e1_len"], wall), flush=True)

if __name__ == "__main__":
    open(LOG, "w").close()
    order = ["champ", "seed", "seed+k", "seed+e1", "seed+k+e1",
             "champ_no_k", "champ_no_e1", "champ_k2", "champ_k234"]
    for name in order:
        run(name, DESIGNS[name])
    print("[probe] DONE", flush=True)
