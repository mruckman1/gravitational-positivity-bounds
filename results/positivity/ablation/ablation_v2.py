"""HONEST ABLATION v2 — conventional search vs the LLM campaign, on the IDENTICAL
task, with two corrections over v1:

  (1) FAIRNESS FIX: v1's mutation operator could never make k_powers (the
      C_2^imp-susy tower) non-empty — a single add from [] hit len==1 and the
      0-or->=2 rule wiped it back to []. Empirically (20k mutations) k_powers
      stayed [] forever, so v1 STRUCTURALLY could not represent the LLM
      champion's key move (k_powers=[2,3,4,5]). v2 lets k_powers jump []->{>=2
      valid block} and grow/shrink while preserving the 0-or->=2 invariant, so
      the full design space the LLM had is reachable by mutation.

  (2) PARITY FIX: the LLM campaign ran under ShinkaEvolve with NO explicit job
      time= set, so it used the runner's EWMA adaptive per-eval limit:
      limit = max(60, 5*EWMA + 60), EWMA (alpha=0.3) over completed eval wall
      times; a job exceeding it is killed as hung and counts as a FAILURE (no
      certificate). v1 had no timeout, so it ran a MORE permissive protocol
      (multi-hour grinders were 'certified' that the campaign would have
      killed). v2 applies the identical EWMA limit via a forked child process.

Everything else is identical to the LLM campaign: same evaluator
(tasks/grav_extu/evaluate.py aggregate(), same certifier, fast_refuse, fitness),
same seed design (gen-0 propose_design), same design-space ranges
(tasks/grav_extu/initial.py), same budget (462 evals).

Historical LLM-arm reference: 462 evals, 175 certified, best 2.96514 @ j_audit=40
(iteration-0 canonical); the winning move populated k_powers=[2,3,4,5] and
e1_powers=[0,2,4,6,8] from empty.

Usage: ablation_v2.py {random|ga} [budget=462] [seed_int=1]
"""
import sys, os, json, time, random, copy
import multiprocessing as _mp

REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
import importlib.util

ARM = sys.argv[1] if len(sys.argv) > 1 else "random"
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 462
SEED_INT = int(sys.argv[3]) if len(sys.argv) > 3 else 1
# PIN_J40: force the matched, headline scope (j_audit=40) so the search must
# confront the same valley the LLM crossed (the seed basis REFUSES at j=40
# until the k/e1 towers are built) instead of drifting to the easy j=36 scope
# that the weak 0.6/step fitness discount under-penalizes.
PIN_J40 = os.environ.get("ABL_PIN_J40") == "1"
random.seed(SEED_INT)


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_abl")
init = _load(os.path.join(REPO, "tasks/grav_extu/initial.py"), "init_abl")
SEED = init.propose_design()
if PIN_J40:
    SEED["j_audit"] = 40
HERE = os.path.dirname(os.path.abspath(__file__))
_TAG = "j40_" if PIN_J40 else ""
LOG = os.path.join(HERE, "ablation_v2_%s%s_s%d.jsonl" % (_TAG, ARM, SEED_INT))

TOWERS = {"f_powers": (2, 16), "h_powers": (0, 10), "e_powers": (0, 8),
          "x4_powers": (0, 8), "x6_powers": (0, 8), "e1_powers": (0, 8),
          "k_powers": (0, 6)}
PMAXES = ["1", "9/8", "5/4", "3/2"]

# ---- EWMA per-eval timeout (parity with ShinkaEvolve runner) --------------
ALPHA = 0.3
_ewma = [None]          # mutable cell
BOOTSTRAP_LIMIT = 1800.0  # generous cap for the very first eval (seed design)


def _limit():
    if _ewma[0] is None:
        return BOOTSTRAP_LIMIT
    return max(60.0, _ewma[0] * 5.0 + 60.0)


def _update_ewma(wall):
    if _ewma[0] is None:
        _ewma[0] = float(wall)
    else:
        _ewma[0] = ALPHA * float(wall) + (1 - ALPHA) * _ewma[0]


def ncols(d):
    return sum(len(d[t]) for t in TOWERS)


def _fix_kpowers(cur, lo, hi):
    """Preserve the 0-or->=2 invariant WITHOUT making the tower unreachable:
    a stray singleton grows to a valid pair (most of the time) instead of
    always collapsing to []."""
    cur = sorted(set(cur))
    if len(cur) == 1:
        cand = [x for x in range(lo, hi + 1) if x not in cur]
        if cand and random.random() < 0.85:
            cur = sorted(cur + [random.choice(cand)])
        else:
            cur = []
    return cur


def mutate(d):
    d = copy.deepcopy(d)
    for _ in range(random.randint(1, 3)):
        op = random.random()
        if op < 0.55:                       # tower edit
            t = random.choice(list(TOWERS))
            lo, hi = TOWERS[t]
            cur = sorted(set(d[t]))
            # k_powers seeding: from empty, jump straight to a valid >=2 block
            if t == "k_powers" and len(cur) == 0 and random.random() < 0.5:
                k = random.randint(2, 4)
                cur = sorted(random.sample(range(lo, hi + 1), k))
                d[t] = cur
                continue
            r = random.random()
            if r < 0.4 and len(cur) > (0 if t not in ("f_powers", "h_powers")
                                       else 1):
                cur.remove(random.choice(cur))
            elif r < 0.8 and len(cur) < hi - lo + 1:
                cand = [x for x in range(lo, hi + 1) if x not in cur]
                if cand:
                    cur.append(random.choice(cand))
            elif cur:
                i = random.randrange(len(cur))
                step = random.choice([-2, -1, 1, 2])
                v = min(hi, max(lo, cur[i] + step))
                if v not in cur:
                    cur[i] = v
            if t == "k_powers":
                cur = _fix_kpowers(cur, lo, hi)
            d[t] = sorted(set(cur))
        elif op < 0.7:
            d["cap"] = float(min(20000, max(50, d["cap"]
                              * 10 ** random.uniform(-0.4, 0.4))))
        elif op < 0.8:
            d["give"] = float(min(0.40, max(0.01, d["give"]
                               * 10 ** random.uniform(-0.35, 0.35))))
        elif op < 0.88:
            d["max_refine"] = random.randint(2, 5)
        elif op < 0.95:
            d["j_audit"] = 40 if PIN_J40 else random.choice([36, 38, 40])
        else:
            d["p_max"] = random.choice(PMAXES)
    if PIN_J40:
        d["j_audit"] = 40
    while ncols(d) > 40:
        t = max(TOWERS, key=lambda k: len(d[k]))
        d[t] = d[t][:-1]
    return d


def crossover(a, b):
    d = {}
    for t in TOWERS:
        d[t] = copy.deepcopy(random.choice([a, b])[t])
        if t == "k_powers":
            d[t] = _fix_kpowers(d[t], *TOWERS[t])
    for k in ("cap", "give", "max_refine", "j_audit", "p_max"):
        d[k] = random.choice([a, b])[k]
    while ncols(d) > 40:
        t = max(TOWERS, key=lambda k: len(d[k]))
        d[t] = d[t][:-1]
    return d


# ---- forked-child eval with hard wall-clock kill --------------------------
_CTX = _mp.get_context("fork")


def _child(d, q):
    try:
        m = ev.aggregate([d])
        q.put(("ok", float(m.get("combined_score", -1e9)), m.get("public", {})))
    except Exception as e:                       # noqa
        q.put(("err", -1e9, {"status": "error", "err": str(e)[:200]}))


_n = 0
_best = (-1e18, None, None)


def score(d):
    global _n, _best
    _n += 1
    lim = _limit()
    q = _CTX.Queue()
    p = _CTX.Process(target=_child, args=(d, q))
    t0 = time.time()
    p.start()
    p.join(lim)
    timed_out = False
    if p.is_alive():
        p.terminate(); p.join(5)
        if p.is_alive():
            p.kill(); p.join()
        timed_out = True
        s = -1e9; pub = {"status": "timeout"}
    else:
        try:
            tag, s, pub = q.get_nowait()
        except Exception:
            s = -1e9; pub = {"status": "no_result"}
    wall = time.time() - t0
    if not timed_out and pub.get("status") not in ("timeout", "no_result", "error"):
        _update_ewma(wall)          # only real completions shape the limit
    rec = {"n": _n, "arm": ARM, "seed": SEED_INT, "score": s,
           "status": pub.get("status"), "c": pub.get("c"),
           "j_audit": d.get("j_audit"), "p_max": d.get("p_max"),
           "k_len": len(d.get("k_powers", [])), "e1_len": len(d.get("e1_powers", [])),
           "wall_s": round(wall, 1), "timed_out": timed_out,
           "limit_s": round(lim, 1), "ewma_s": round(_ewma[0] or 0, 1),
           "design": d}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if s > _best[0]:
        _best = (s, d, pub)
        print("[%s.s%d %d/%d] NEW BEST score=%.4f status=%s c=%s ja=%s pm=%s "
              "k=%d e1=%d" % (ARM, SEED_INT, _n, BUDGET, s, pub.get("status"),
                              pub.get("c"), d.get("j_audit"), d.get("p_max"),
                              len(d.get("k_powers", [])),
                              len(d.get("e1_powers", []))), flush=True)
    elif _n % 20 == 0:
        print("[%s.s%d %d/%d] best=%.4f ewma=%.0fs lim=%.0fs" %
              (ARM, SEED_INT, _n, BUDGET, _best[0], _ewma[0] or 0, lim),
              flush=True)
    return s


print("=== ABLATION v2 arm=%s seed=%d budget=%d (corrected operator + EWMA "
      "timeout; identical evaluator to the LLM campaign) ===" %
      (ARM, SEED_INT, BUDGET), flush=True)

if ARM == "random":
    cur = copy.deepcopy(SEED); cur_s = score(cur)
    stall = 0
    while _n < BUDGET:
        cand = mutate(cur)
        s = score(cand)
        if s > cur_s:
            cur, cur_s, stall = cand, s, 0
        else:
            stall += 1
            if stall >= 60:
                cur, cur_s, stall = copy.deepcopy(SEED), -1e18, 0
elif ARM == "ga":
    POP = 10
    pop = [(score(SEED), copy.deepcopy(SEED))]
    while len(pop) < POP and _n < BUDGET:
        d = mutate(SEED); pop.append((score(d), d))
    while _n < BUDGET:
        pop.sort(key=lambda t: -t[0])
        elite = pop[:2]

        def pick():
            a, b = random.sample(pop, 2)
            return a if a[0] >= b[0] else b
        child = crossover(pick()[1], pick()[1])
        if random.random() < 0.8:
            child = mutate(child)
        pop = elite + sorted(pop[2:], key=lambda t: -t[0])[:POP - 3] \
            + [(score(child), child)]
else:
    raise SystemExit("arm must be random|ga")

print("\n=== ARM %s seed %d FINAL: best score %.4f ===" %
      (ARM, SEED_INT, _best[0]), flush=True)
print(json.dumps({"best_score": _best[0], "best_public": _best[2],
                  "best_design": _best[1]}, indent=1), flush=True)
