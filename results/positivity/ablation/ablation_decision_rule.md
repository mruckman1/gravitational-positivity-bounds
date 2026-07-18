# Ablation decision rule (pre-registered before the fleet reports)

## Purpose
Decide whether "Machine-Guided" in the paper title is earned: does an LLM-guided
search reach a frontier that conventional (non-LLM) search of the same budget
does not? The paper body already disclaims LLM *superiority*; this decides the
honest framing of the title.

## Reference (the LLM campaign, ledgered)
- 462 evals, 175 certified (38%), best certified **2.96514** @ j_audit=40.
- Winning move: populated `k_powers=[2,3,4,5]` (C2^imp-susy tower) and
  `e1_powers=[0,2,4,6,8]` (E1 null) from EMPTY — an iteration-0 canonical design.

## Conventional fleet
- 12 independent seeds: 6 random hill-climb + 6 GA.
- IDENTICAL evaluator / seed design / design-space ranges / budget (462).
- CORRECTED operator: k_powers now reachable (v1 could not populate it — proven
  over 20k mutations; v1's result is INVALID and discarded).
- EWMA per-eval timeout = max(60, 5*EWMA+60), alpha=0.3 — parity with the
  campaign's ShinkaEvolve runner (which set no explicit job time=).

## Decision (evaluated at checkpoints; report the budget reached)
- **RETITLE toward "certified functional design"** (LLM edge NOT demonstrated):
  any conventional seed certifies <= 2.969 within budget, OR pooled conventional
  best <= 2.970.
- **"Machine-Guided" DEFENSIBLE** (with the ablation reported in-paper): after
  each seed >= 120 evals AND pooled budget >= 2x462, conventional best certified
  stalls >= 2.99, AND the conventional arms DID explore champion-like structure
  (k_powers populated in a meaningful fraction of designs) — so the gap is a real
  search-difficulty gap, not a reachability artifact.
- **INCONCLUSIVE** (pooled best in (2.970, 2.99)): report honestly; default to
  the conservative framing (retitle), since the body already disclaims
  superiority.

## Confound guards (report regardless of outcome)
1. **k-population rate**: fraction of evaluated conventional designs with
   k_powers >= 2. If low, check WHY — does populating k rarely improve fitness
   (a valley greedy search won't cross)? That itself is a legitimate reason
   machine guidance helps, and must be stated, not hidden.
2. **timeout rate**: fraction of evals killed by the EWMA limit (parity check —
   should be a small tail, matching the campaign's hung-job kills).
3. **certified rate**: compare to the LLM's 38%. A conventional arm that
   certifies MORE but WORSE (easy high values) vs the LLM's fewer-but-lower is
   the expected "greedy finds easy wins, LLM makes the leap" signature.
4. Report per-arm (random vs GA) separately, and pooled.
