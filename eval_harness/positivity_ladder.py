"""Phase-1 completion: the CHVD convergence ladder with the FULL generated
null-constraint basis. gt3 lower at null_order n = 6,8,10,12,14,16 (Table 2
targets: -10.3662 at n=6 -> -10.3465 at n=16) and the gt4 box at n=16
(expect [0, 1/2], upper exact). J_max escalates on audit failure — the audit
is the soundness gate, so an insufficient J_max can only cost retries, never
a spurious certified bound. Resume-safe JSONL."""
import json
import time
from pathlib import Path

from qgse.verifiers.positivity import (PositivitySpec, PositivityVerifier,
                                       ensure_nulls)

OUT = Path("results/positivity/ladder.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        try:
            _d = json.loads(line)
            if "error" not in _d:
                done.add(_d["label"])
        except Exception:  # noqa: BLE001
            pass

print("generating null basis to n=16 ...", flush=True)
t0 = time.time()
names = ensure_nulls(16)
print(f"  {len(names)} nulls in {time.time()-t0:.0f}s: {names}", flush=True)

V = PositivityVerifier()
JOBS = ([(f"gt3_lower_n{n}", dict(target="gt3", side="lower", null_order=n))
         for n in (6, 8, 10, 12, 14, 16)]
        + [("gt4_upper_n16", dict(target="gt4", side="upper", null_order=16)),
           ("gt4_lower_n16", dict(target="gt4", side="lower", null_order=16))])

for label, kw in JOBS:
    if label in done:
        print(f"[skip] {label}", flush=True)
        continue
    t0 = time.time()
    rec = {"label": label}
    for j_max in (100, 200, 400, 800):
        try:
            r = V.extremal_bound(PositivitySpec(j_max=j_max, j_audit=200,
                                                **kw))
            rec.update({k: r[k] for k in ("bound", "bound_exact", "statement",
                                          "sdpb_objective", "scope", "j_max")})
            rec["n_nulls"] = len(r["nulls"])
            rec["tail_proven"] = r["audit"]["tail_proven"]
            print(f"[{label}] {r['statement']} (J_max={j_max}, "
                  f"{time.time()-t0:.0f}s) scope: {r['scope'][:60]}",
                  flush=True)
            break
        except RuntimeError as e:
            if "audit FAILED" in str(e) or "spurious" in str(e):
                print(f"[{label}] audit failed at J_max={j_max} — escalating",
                      flush=True)
                continue
            rec["error"] = str(e)[:200]
            print(f"[{label}] ERROR {str(e)[:120]}", flush=True)
            break
    else:
        rec["error"] = "audit failed at all J_max up to 800"
        print(f"[{label}] EXHAUSTED J_max escalation", flush=True)
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
print("ladder done", flush=True)
