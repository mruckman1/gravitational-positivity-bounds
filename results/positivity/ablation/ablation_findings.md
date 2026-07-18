# Ablation findings (accumulating; corrected v2)

## The k-valley, measured directly (probe, all j_audit=40 unless noted)
| design | k | e1 | result @ j=40 |
|---|---|---|---|
| seed | 0 | 0 | certified 3.061 (at j=36 default) |
| seed + k only | 4 | 0 | **lp_infeasible** |
| seed + e1 only | 0 | 5 | certified 3.002 |
| seed + k + e1 | 4 | 5 | **certified 2.989** |
| champ (full LLM design) | 4 | 5 | lp_infeasible (LP re-derivation noise) |
| champ − k | 0 | 5 | **lp_infeasible** (k-tower essential) |
| champ − e1 | 4 | 0 | certified 3.175 (much worse without e1) |
| champ, k=[2,3] | 2 | 5 | timeout (grinds) |
| champ, k=[2,3,4] | 3 | 5 | **lp_infeasible** (needs FULL k=[2,3,4,5]) |

**Reading:** the beneficial j=40 structure is a JOINT, correctly-sized addition of
the k (C2-susy) AND e1 (E1) towers. Individually the pieces are neutral-or-broken:
k alone -> infeasible; truncated k -> infeasible; champion minus k -> infeasible;
minus e1 -> 3.175. Strong epistasis. The full joint move (seed+k+e1) reaches 2.989
@ j=40 -- only 0.024 above the LLM champion's 2.96514 (which adds further h/e/x4/x6
restructuring). So the LLM's edge is "reliably makes the joint structural leap,"
NOT "reaches something unreachable."

## Fleet, j=40 PINNED (matched scope), 72 evals so far
- 32% certified; best 3.012 (gap +0.047 to LLM); 36 audit_failed, 13 infeasible.
- Only 8% of evals even TRY k>=2; those fail (best k-populated cert = 4.40).
- The search wanders the refusal PLATEAU (partial towers refuse -> flat fitness
  0.5, no gradient), rarely assembling a full tower. It plateaus ~3.01 with
  k-empty designs, does NOT make the joint move in this budget.

## Fleet, FREE-j (secondary), 82 evals
- Drifts to easy j=36 (weak 0.6/step scope discount). Best 2.948 @ j=36.
- At narrow scope, conventional MATCHES the LLM (LLM's narrow-scope was 2.9556@j38,
  not promoted). No LLM edge at narrow scope.

## Emerging conclusion (pending more j=40 budget)
- Narrow scope: conventional == LLM (no edge).
- Headline j=40 scope: conventional plateaus ~3.01 (gap ~0.05); the beneficial
  structure is a joint epistatic tower move greedy/GA navigate poorly. BUT the
  move is reachable in principle (2.989 certifies), so the LLM edge is modest and
  about RELIABLY making the structural leap, not accessing the inaccessible.
- => Title: "Machine-Guided" superiority is NOT strongly supported. Conservative
  framing ("certified functional design", LLM as mutation operator) is the honest
  call -- consistent with the paper's existing in-body disclaimer. Confirm with
  more j=40 budget (does any seed make the joint move & reach <=2.99?).
