# §7 Ablation: does the language model help?

Controlled ablation for the paper's Section 7 — LLM-guided search vs conventional
(random-restart hill-climb + genetic algorithm) on the **identical** evaluator,
seed design, design-space ranges, per-eval budget (462), and per-eval EWMA
wall-clock limit. Only the mutation operator differs.

**Result (matched j_audit=40 headline scope):** the best of 12 conventional runs
certifies `g_0 <= 2.96597`, gap `+8e-4` to the LLM's `2.96514` — an effective tie;
most seeds plateau ~3.01–3.03. At the narrower (free j_audit) scope conventional
matches the LLM. **No demonstrated LLM search advantage** → the title is "Automated
Search" and the LLM is reported as the search's mutation operator.

## Files
- `ablation_v2.py` — the harness (arm ∈ {random, ga}, seed, budget). Corrected
  operator (k/C₂-susy tower reachable — the v1 in `../drivers/` could not build it
  and its result is invalid) + forked-child EWMA per-eval timeout for parity with
  the ShinkaEvolve campaign. Run pinned to j=40 with `ABL_PIN_J40=1`.
- `kvalley_probe.py` — direct probe of the C₂-susy + E₁ tower fitness landscape
  (the epistatic valley: towers jointly beneficial, individually infeasible/neutral).
- `pool_j40.py` — pools the matched-scope (j=40) fleet; prints the decision.
- `pool_ablation.py` — pools the free-scope (secondary, narrow-scope) fleet.
- `launch_fleet.sh`, `gate_launch.sh` — orchestration used to run the 12-seed fleet.
- `ablation_decision_rule.md` — the **pre-registered** decision rule (written before
  the fleet reported).
- `ablation_findings.md` — the results writeup incl. the measured valley table.
- `data/ablation_v2_j40_{random,ga}_s{1..6}.jsonl` — the 12 primary j=40 run logs.
- `data/kvalley_probe.jsonl` — probe results.
- `data/freej_logs/` — the free-scope (narrow) secondary run logs.

## Reproduce the analysis (no compute; reads the shipped logs)
```
.venv/bin/python results/positivity/ablation/pool_j40.py        # matched-scope verdict
.venv/bin/python results/positivity/ablation/pool_ablation.py   # narrow-scope secondary
```

## Re-run the fleet (heavy: hours; needs shinka_fork + LLM-free, pure local search)
```
bash results/positivity/ablation/launch_fleet.sh 462   # 12 seeds; or run one:
ABL_PIN_J40=1 .venv/bin/python results/positivity/ablation/ablation_v2.py random 462 1
```
Each eval is a full LP + exact audit (minutes); the fleet is multi-hour. See
`ablation_findings.md` for the two fairness bugs (unreachable k-tower; missing EWMA
timeout) and the scope confound that were caught and corrected relative to v1.
