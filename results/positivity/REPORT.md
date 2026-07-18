# 4D positivity-bounds verifier — Phase 0 recon + Phase 1 core (2026-07-04)

**The pivot**: the certified-search architecture repointed from the AdS3 toy
at real 4D physics — dispersive positivity bounds on EFT Wilson coefficients
(EFT-hedron line; theorems from analyticity + unitarity + Regge boundedness).
New verifier family `qgse/verifiers/positivity.py`; same discipline: grounded
SDP disposes, certificates or it didn't happen, validate-known-before-unknown.

## Phase 0 recon (5-agent workflow, adversarially verified)
- **Targets pinned** (all CONFIRMED by independent cross-check): Adams et al.
  c_3 > 0; CHVD 2011.02957 conventions + the validation ladder:
  analytic Cauchy-Schwarz [-16, 3]; single-null EXACT optimum
  gt_3 >= -(9/2 + (7/4)sqrt(61/5)) = -10.612487218800544 (eq 3.12); headline
  d=4 box -10.346 <= gt_3 <= 3, 0 <= gt_4 <= 1/2 (eq 4.2). Phase-2 target:
  CMRS 2102.08951 gravitational bounds.
- **Two adversarial catches before any code**: (1) n_5(J=4) = +3600/m^10, not
  +8400 (extraction typo); (2) **J-truncation + Regge block is UNSOUND** —
  explicit counterexample: J_max=2 functional "certifies" gt_3 >= -10.179
  while a legal 2-state spectrum sits at -10.612; the functional is negative
  at the excluded spin J=4. Bounds require an a-posteriori exact audit.
- **Backend**: reuse pmp2sdp+SDPB (natively a PMP; certificates need
  arbitrary precision; parsing/cleanup infra transfers verbatim).
- Recon also found the published eq-(2.16/18) kernel inconsistent with
  eqs (2.15)+(2.22) by one power of m^2 — resolved symbolically.

## The verifier (what makes it sound)
- Every SDP row DERIVED symbolically at import from the master dispersion
  kernel and asserted against closed forms (g_2=<1/m^4>, g_3=<(3-2X)/m^6>,
  n_4=2X(X-8)/m^8, n_5=2X(2X^2-43X+150)/m^10; X=J(J+1)) — import fails on
  any mismatch. My first kernel had the extra-m^2 trap; the assert caught it.
- SDP: one 1x1 PMP block per even spin J <= j_max + the exact Regge block
  q(xi) = lim rows(x, X=xi(1+x))/(1+x)^{k_max-2} (lossless).
- **Soundness gate**: SDPB's functional is rounded to rationals with a
  margin backoff in the WEAKENING direction, then re-verified in exact QQ
  arithmetic: per-spin Sturm positivity to j_audit, plus (deg_X=2) a COMPLETE
  tail proof (disc_X < 0 beyond the Cauchy bound + strip spin cutoff) — the
  certified bound is then a theorem independent of SDPB numerics. deg_X>=3
  tails: honestly SCOPED to "all spins J <= j_audit" (never silent).
- 3-way discipline: optimal-with-audit / scoped / hard error.

## Validation (eval_harness/positivity_check.py): 22/22 PASS
Unit (exact sympy): kernel-derived rows; n_J^(4)=16pi(2J+1); P_J Taylor
formula; corrected n_5 spots; Regge blocks as exact limits; analytic-optimum
root structure (J=2 root at x=0, J=4 double root at 37/(2a*)); odd-spin
negativity (even-only mechanism); exact 2-state witness closing strong
duality; **audit REFUSES the known-spurious J_max=2 functional and ACCEPTS
the true one with complete tail proof**; scalar-exchange end-to-end.
SDPB: gt_3 lower (n4) = -10.612487219 (matches analytic to 1e-8; certified
-10.612487229, margin on the sound side, ALL-SPIN exact certificate);
gt_3 upper = 3.000000000 exact; (n4,n5) matches CHVD Table 2 n=5.

## Next
Phase 1 completion: X_k(t)-generated null-constraint basis (Mandelstam
n = 6..16) -> reproduce the converged -10.346 and gt_4 in [0, 1/2]; general
deg_X tail certificates. Phase 2: spin + graviton pole (CMRS). Phase 3: LLM
on dispersive-functional construction (nulls/arcs/kernel design space) vs
literature + greedy baselines — the methodology experiment on 4D physics.

## Phase 1 COMPLETE: full null basis + convergence ladder (2026-07-04)

**Null-constraint generator** (`ensure_nulls`): derives the complete crossing
null basis at any Mandelstam order from the residue identity — g^2/lambda
pole-leakage columns included in the nullspace (killed, not assumed absent),
trivial-symmetry combos discarded, homogeneity asserted. Validated: exact
n4/n5/n6 reproduction + literature counts (1,1,1,2,2,2,3 at n=4..10);
35 constraints at n<=16 in 23s.

**Convergence ladder** (every rung an exact per-spin certificate to J<=200,
margin-weakened on the sound side; deg_X>=3 tails scoped, not assumed):

| null order n | certified gt_3 >= | CHVD Table 2 |
|---|---|---|
| 6  | -10.366197 | -10.3662 (exact match) |
| 8  | -10.349203 | — |
| 10 | -10.347652 | — |
| 12 | -10.347288 | — |
| 14 | **-10.346862** | -> -10.3465 converged |
| 16 | -10.356950 (SUBOPTIMAL, see below) | — |

**gt_4 box: [-5e-32, 0.5] — reproduces CHVD's exact [0, 1/2]** (upper
saturates the spin-0 threshold amplitude; lower's -5e-32 is the sound-side
margin artifact of exactly 0).

**Soundness architecture validated in production, twice:** (1) the n>=12
rungs initially REFUSED to certify at every J_max — the audit catching my
nsimplify tolerance-rounding corrupting 20+-component functionals; the bug
produced refusals, never a wrong number (the designed asymmetry). Fix: exact
mpf->rational conversion (binary mantissa/exponent, zero tolerance).
(2) The n=16 rung converged to a LOOSER value than n=14 (monotonicity
violation = SDPB suboptimality at 768 bits with 37 components) — the audit
still passed because the weak bound is TRUE; suboptimality is honest, only
spuriousness is fatal. High-precision redo (1152 bits) queued.

**Status: the verifier reproduces the complete CHVD scalar program** —
the analytic optimum, the convergence ladder to ~1e-4 of the converged
value, and both exact box endpoints. Remaining engineering: deg_X>=3 tail
certificates; n=16 conditioning. Next physics: the graviton pole (Phase 2).

## Phase 2 recon COMPLETE (2026-07-04): the graviton formulation, verified

Full artifacts: `results/positivity/phase2_recon/`. All claims CONFIRMED by
independent adversarial recomputation. The essentials:

**Structural gift**: CMRS's C_{k,u} sum rules ARE our validated B_k residue
machinery (t renamed u, same subtraction points). The graviton hits EXACTLY
ONE cell: (k=2, Laurent r=-1) with coefficient -8piG (piece-split -1,-1,+1
across st/u, su/t, tu/s — sympy-verified); lambda_3^2/lambda_4 are
dispersively invisible; all k>=4 Taylor cells are G-free so the null tower
survives verbatim. The (2,-1) cell is off the Taylor lattice => C_2-family
functionals must be SMEARED: f(p) on u=-p^2, p in (0,M], with the improved
kernel C_2^imp (eq 44, geometric resummation verified, regular at u=-m^2)
and the m->inf Bessel closure fhat(b)>=0.

**Soundness ledger** (per approximation step): basis truncation SOUND;
J-truncation UNSOUND (same class as our counterexample); m-grid + adaptive
refinement UNSOUND (CMRS's own Fig 8 shows residual -1.8e-10 dips — their
refinement mitigates, never proves); large-b 2x2 PSD block SOUND as a
restriction. Our replacement for grids: per-spin EXACT audit on the full
continuum — integer-power smearing integrals are rational + atan/log in
w=1/m; sandwich the transcendentals with alternating-series rational bounds
and decide by Sturm on (0,1]. The audit, not the solver, is the certificate.

**Validation targets pinned** (Table 3, units 8piG=1, M=1): D=7 rays
g_2 - g_3/3 + 18.0717 >= 0 and g_2 + 0.129850 g_3 + 2.75853 >= 0
(17-dim functional space, J_max=42, ~1e-4 converged); no-gravity anchor
kappa(D) closed form (eq 34-35); D=6 min g_2 = -9.73 (17-fn) / -9.57
(92-fn); SUSY D=10 R^4: 0 <= g_0 <= 3.000*8piG/M^6 vs string 2*zeta(3).
Six traps, each with an exact computable test (the -1,-1,+1 pole split;
u-sign flip; lambda_3-vs-G confusion; p-space-vs-b-space positivity
[f=p^4>=0 has fhat<0 at b~7 in D=7 — the audit must reject it]; D=4 rows
smuggled into D>4; Taylor/J_max order-of-limits).

**Phase-2a defined**: D=7, certify g_2 + alpha g_3 M^2 >= -c_cert 8piG/M^2
on the published rays — the first certified 4D-style gravitational number
(new module `gravity_positivity.py`; general-D layer; assertion battery
first; ~10-14 focused days estimated by the recon).

## PHASE-2a v1: FIRST CERTIFIED GRAVITATIONAL BOUND (2026-07-05)

**CERTIFIED (smoke settings, j_max=6):** for any UV-completable scalar+gravity
EFT in D=7 with spectrum spins J <= 6:
  g_2 - 0.333033 g_3 M^2 + 35.9209 * 8piG/M^2 >= 0
(exact rational certificate in results/positivity/grav_smoke6.log; audit
passed iteration 0). Weak vs CMRS 18.0717 by design — the v1 basis has no
null smears and the coefficient cap trades tightness for certifiability
(measured frontier: cap 200 -> 35.6, cap 1000 -> 26.6, uncapped 21.7 =
uncertifiable degenerate functional).

**The architecture that got here** (each mechanism validated by catching a
real defect): sound exact smearing integrator (caught sympy returning a
WRONG symbolic-parameter antiderivative: -56041 vs true +3.566 — verified
by quadrature); two-regime exact audit (interval arithmetic PROVEN unusable
by profiling — 3.36M boxes from the dependency problem — replaced by
Taylor-sandwich Sturm below w=1/2 and segment-enclosure Sturm above);
certifiability-first LP (capped coefficients + slack maximization: the
uncapped optimum saturates constraints and is un-auditable BY DESIGN);
interior-ray inset (the audit correctly REFUSED a functional whose
rationalization violated the saturated J=0/m=1 corner at the 1e-6 level).

Production runs (j_max=42, j_audit=42, both Table-3 rays) launched.

## PHASE-2a COMPLETE: production certified gravitational bounds (2026-07-07)

**The certified trio** (all exact rational certificates, D as stated,
spins J <= 40 scope, units M=1):
1. g_2 + 0.129808 g_3 M^2 + 2.9388 * 8piG/M^2 >= 0   (D=7)
   — WITHIN 6.5% of CMRS's hand-tuned 2.75853, certified to a stronger
   standard than the published number (see knife-edge note).
2. g_2 - 0.307708 g_3 M^2 + 35.4057 * 8piG/M^2 >= 0  (D=7)
   — valid but far from the published 18.07 at slope -1/3: the max-g_3
   edge is the genuinely hard region (reserve blend delta=0.064 moved the
   ray); PRIMARY Phase-3 target.
3. g_0 <= 4.3307 * 8piG/M^6                           (D=10 susy sector)
   — the extremality ceiling (CMRS 3.000; type II string at 2 zeta(3) =
   2.4041); extra axiom s^2 M -> 0 flagged in scope.
Plus the v1 bound (c=35.92 at slope -0.333, J<=6) as the first-ever entry.

**Two findings about the field's own numbers, surfaced by our standard:**
- KNIFE-EDGE: the exact constraint set at J=42 (fine threshold sampling)
  is marginally INFEASIBLE in the 16-column basis — CMRS's own published
  functionals violate it at the 1.8e-10 level (their Fig 8), absorbed by
  solver tolerance. Our pipeline surfaced it as hard infeasibility; our
  certificates scope to J<=40 instead of hiding in tolerance.
- TANGENCY IS PHYSICS: LP-optimal functionals touch zero on continuum
  families (near-threshold high-spin rows = large-b Bessel rows in
  disguise, b=2J/m); certification requires either margin purchased
  explicitly or infinite refinement. The POSITIVITY RESERVE architecture
  (Chebyshev-center max-margin functional of the basis, blended at
  delta ~ 1e-3, escalating only on audit failure) decouples tightness
  from auditability at priced cost — no naive positive profile works
  (the J=2 threshold row forces the LP's sign-mixed structure).

**Build ledger for the arc** (each failure structural, none a wrong
number): sympy symbolic-parameter integrate WRONG (quadrature-verified);
interval-arithmetic dependency death (profiled, 3.36M boxes); Bernstein
certificates 170x over Sturm; heuristic-GCD ring-path failure (patched
deterministic); float overflow at high-J/large-m (finite-guards);
quadrature-vs-exact refinement stall (exact feedback rows); tool-timeout
process-group kills (session detachment); slack pinned by structurally-
zero Bessel rows (channel separation); sliding-dip chases (spread pins);
reserve blend (final architecture). Audit passed only genuine functionals
throughout — zero spurious certificates across ~15 attempts.

**Next: Phase 3** — LLM smearing design. Two arenas with measured
baselines: close ray-1's 35.4 -> 18.07 gap; push the g_0 ceiling
4.33 -> 3.00 -> toward 2.4041 (the string-extremality question).

## PHASE 3 COMPLETE: LLM-designed dispersive functionals (2026-07-13)

Two evolutionary campaigns (ShinkaEvolve fork; Claude-5-family LLM at
`verbosity=max` via OpenRouter; certified-or-nothing fitness — audit
refusals score BELOW any certificate). Both harvested at convergence;
both champions RE-AUDITED standalone outside the evolution harness and
reproduce their campaign certificate BIT-FOR-BIT (exact-rational delta
0.00e+00). Artifacts: `results/positivity/artifacts/*.json`; ledger
entries `g0_arena1b_champion`, `hardedge_arena2_champion` in
`grav_rays.jsonl`. Combined LLM spend: ~$31.49.

### Arena-2 (D=7 hard edge, max-g_3 ray) — a genuine evolutionary descent
- **Certified c = 23.172254454160509** (exact rational
  118449254407702708994395974385009/5111684520900619925696652238137),
  slope certified exactly -0.33299764
  (-1134785915216052803784694388747/3407789680600413283797768158758).
  Scope: UV-completable scalar+gravity EFTs in D=7, ALL spins J<=40,
  exact continuum audit, Bessel closure. Reproduces via 1 refine
  iteration standalone.
- **The learning curve is real**: 36.679 (gen-0 certified start) ->
  29.043 -> 23.828 -> 23.517 -> 23.178 -> ... -> 23.1722 — a 15-point
  improving trajectory (gen-0 + 14 strictly-improving certified steps).
  The winning genome mechanically reconstructs the best-known design from
  a self-maintained provenance ledger (`LINEAGE`), then takes one
  isolated, evidence-backed delta_base shrink — an anti-regression
  discipline the LLM invented after a mid-campaign ledger-omission bug
  silently regressed the reconstruction (documented in the champion's own
  `STRUCTURAL_RISK_NOTES`).
- 152 evals with metrics (of 223 gen attempts): 81 certified, 67
  LP-infeasible, 4 error/null; 71 attempts produced no metrics. Config:
  C_2^imp powers 2-8, X_4 nulls 0-6, X_6 nulls 0-7, refine 6, delta_base
  ~2.34e-5 (2.85e-5 lineage anchor + 1 auto-shrink; reserve blend),
  p_max=1, j_audit=40.
- **HONEST VERDICT**: 35% tightening of the program's weakest bound,
  closing **71%** of the seed->published gap (35.41 -> 23.17 of the way
  to 18.0717) — but it did **NOT cross the published 18.0717**. The
  published number rests on the tolerance-absorbed standard (see the
  J=42 knife-edge, Phase-2a); our 23.17 is a theorem under exact
  continuum audit. Relative win, not an absolute record.

### Arena-1b (D=10 SUSY R^4 extremality ceiling) — design-space floor confirmed
- **Certified g_0 <= 3.013573408191174 * 8piG/M^6** (exact rational
  723814389000578626807240249608827159491797563702048/
  240184754429138374183295825629012011327022547950765). Scope: D=10
  susy-reduced gravity, J<=40, s^2 M->0 Regge axiom. Certified at
  iteration 0; reproduces standalone.
- **Design-space convergence certificate**: the champion value 3.013573
  was independently re-derived by **6 distinct certified evals** (gens 0,
  58, 72, 132, 133, 148); the next-best certified functional sits a hair
  above at 3.0135978. The gen-0 seed (the round-2 champion) was already
  the floor — the campaign did not improve on it, it CONFIRMED it from
  multiple independent lineages.
- 155 evals with metrics (of 176 gen attempts): 48 certified, 11
  LP-infeasible, 96 audit-refused; 21 attempts produced no metrics.
  Config: C_-2 powers f=[2,3,4,6,9,12,16], C_0^imp h=[0,2,4,7,10],
  crossing nulls e/x4/x6, cap 19700, give 0.0271, refine 5, p_max=1
  (extended-u levers available but the p_max=1 basin won).
- **HONEST VERDICT**: 0.45% above CMRS's hand-tuned 3.000 (string
  2 zeta(3) = 2.4041). The evolutionary search mined the design space
  convincingly (six-fold re-derivation) but the p_max=1 basis floor sits
  just above the CMRS number; the sub-3.000 breakout would require the
  extended-u domain (see the standalone 3.0495 at p_max=9/8, J<=36 —
  under the meromorphy+discreteness axiom of CMRS App. B AND the s^2 M->0
  Regge axiom) or a richer basis. No spurious sub-CMRS certificate was
  ever produced.

### The methodology result (what Phase 3 actually demonstrated)
Across **129 certified evals** (48 + 81) over two campaigns and hundreds
of refusals, **zero spurious certificates** — every failure was an audit
refusal or LP infeasibility, never a wrong theorem. Under the exact-audit
standard (no tolerance games), LLM-evolved dispersive functionals
reliably converge to their design space's **true certified frontier**:
Arena-2 by genuine multi-step optimization, Arena-1b by independent
re-derivation of the floor. They land NEAR — but, under our stronger
standard, not below — published optima that rest on solver tolerance.
The automated-functional-design methodology is validated on real 4D
gravitational physics; the remaining gap to string/CMRS numbers is a
statement about certification standards and basis/domain reach, not about
the search.

### Post-campaign all-spin viability upgrade: HONEST FAIL (refusal, not refutation)
The Arena-1b champion was re-run under `viability_rows=True` — Bessel grid
densely extended to b=160 plus the asymptotic-sign row A(b) -> 16 a_2/b^3,
a strictly stronger all-spin standard than the pipeline that produced
3.0136. **It did NOT certify** (~2.5 h, 5 refine iterations,
RuntimeError "no certifiable functional"). What happened, precisely:
- The viability-augmented LP IS feasible: it finds a functional at
  c ~= 3.024 (iter 0: 3.0234, converging to 3.0240) — slightly ABOVE
  3.0136, exactly as the extra constraints predict.
- But the EXACT audit REFUSES it: the small-w alternating-series sandwich
  at the K=26 ladder is "unprovable" at spins J=16, 18, 20, 40 — the
  known coefficient-scale wall (the viability functional's large
  coefficients outrun the K=26 sandwich resolution, same class as the
  J=38/40 extended-domain wall).
- **This is a refusal, not a refutation.** The audit could NOT PROVE
  positivity at those spins; it did NOT prove the functional negative.
  The bound may well be true at the b=160 standard — we simply cannot
  certify it with the current audit resolution. Zero spurious
  certificates: the machinery again failed safe.
**Verdict**: the certified g_0 <= 3.0136 stands at its stated scope (the
b->hi baseline standard, which certifies cleanly). Upgrading it to the
b=160 all-spin standard is blocked on audit resolution, not on the
physics — it needs the higher-K / coefficient-scale-immune small-w bound
already queued (the same unlock that opens J=40 audits on extended
domains). No ledger entry is written for a bound we cannot certify.

## PERFORMANCE + a CORRECTED DIAGNOSIS (2026-07-13, post-harvest)

Two non-LLM engineering results, both regression-gated to BYTE-IDENTICAL
champion certificates (SUSY g_0 = 3.013573408191174 reproduced 4x; D=7 ray
c = 23.172254454160509 and slope -0.33299764 reproduced exactly).

**Speed (get more results faster, zero quality cost):**
- **Kernel caching** (`functools.lru_cache` on the symbolic kernel builders
  `_kernel_expr`/`_xk_kernel`/`_xk_expr`/`cm2_expr`/`c0imp_expr`/
  `c2imp_susy_expr`/`e1_expr` + per-instance `_columns` memo). Profiling
  showed 95% of certify time was REDUNDANT symbolic kernel construction:
  each kernel was rebuilt once PER smear-power inside `_columns` and again
  in both `rows_grid` and `audit`, every refine iteration (294 `_xk_expr`
  calls for 21 unique (k,J,D) keys at j_max=12). Pure functions of small
  integer args => memoizing is byte-identical. Measured: **reduced certify
  25.6x (1587s->62s); full SUSY champion ~9x (1835s->204s); D=7 ray ~3x
  (1155s->361s)**. Rows proven bit-identical cache-warm vs cache-cleared.
- **`_linprog_retry`** on all four HiGHS call sites. HiGHS intermittently
  returns "Status 0: Not Set" on the dense ray LPs — a NON-completion that
  is nondeterministic across process invocations (the identical problem
  solves cleanly in a quiet process; it recurred under the memory pressure
  of many concurrent diagnostic jobs, and hit the caching-only build too, so
  it is orthogonal to any code change). A completed HiGHS solve of identical
  input is deterministic, so a retry is result-preserving; it prevents a
  flaky non-completion from aborting a multi-hour certification.

**CORRECTED DIAGNOSIS (supersedes the "audit resolution" framing in the
viability + extended-u notes above).** The recurring "small-w exact sandwich
unprovable at K=26" refusals at high spin were investigated by directly
measuring min_w E(w) on (0,1] for the extended-u p_max=9/8 iteration-0
functional (c=3.0495):
- J=36: min E = **+0.084** (positive margin -> certifies; this is the J<=36
  scope of the ledgered 3.0495 bound).
- J=38: min E = **-0.028**  (genuinely NEGATIVE).
- J=40: min E = **-0.133**  (genuinely NEGATIVE).
So the "wall" is **NOT audit weakness** — the tight functional genuinely
VIOLATES positivity at J>=38, and the audit was refusing CORRECTLY. No
audit improvement (higher K, sign-resolved brackets, scale-immune
certificates) can certify a functional that is actually negative. The
J<=36 scope of the 3.0495 bound is therefore HONEST and correct, not an
artifact. A scale-immune small-w audit (Pass-2 sign-resolved brackets) was
built and proven sound (E_lb <= E to 60 digits) but then REVERTED: it
targets a non-problem here and could perturb refinement trajectories on
refining runs; the shipped audit is provably the validated original.
IMPLICATION: the path to a tighter extended-u bound (toward the CMRS 3.000
and string 2zeta(3)=2.4041) is a RICHER BASIS that stays positive through
J=40 — not a better audit. The b->160 viability-upgrade refusal is almost
certainly the same functional-violation mechanism (same signature; not
separately min-E-measured). Correction: those bounds are basis/margin
limited, not audit limited.

## EXTENDED-U CAMPAIGN: first certified SUB-CMRS bound (2026-07-14)

Acting on the corrected diagnosis above (the extended-u J>=38 wall is
functional VIOLATION, a basis limit), an LLM campaign was run to find a
RICHER basis that stays positive out to J=40 in the extended-u domain.
Task `tasks/grav_extu/` (seed = champion columns at p_max=9/8, J<=36 = 3.0495;
sys_msg + hint corrected to "add columns", not "narrow j_audit"); $50 budget,
462 evals, Claude-5 at verbosity=max; opt-in refuse-only fast audit path so
the many J=40-violating designs reject in ~0.2s not ~45min.

**RESULT (ledger `g0_extu_campaign_j40`, artifact
`results/positivity/artifacts/extu_champion_j40.json`):**
  g_0 <= 2.965141246567263 * 8piG/M^6
  exact = 45717172277030185398530432103726673602025060928977418959/15418210626544973716286294842876935012187538185570825216
  scope: EXTENDED-u p_max=9/8 (meromorphy+discreteness axiom, CMRS App. B);
         D=10 susy, spins J<=40; s^2 M->0 Regge axiom.
- **Below CMRS's hand-tuned 3.000** — the first certified sub-CMRS bound in
  this sector, at full J<=40, under the exact-audit standard, in territory
  CMRS left as future work. Beats the p_max=1 floor 3.0136 and the seed 3.0495.
- **ITERATION-0 canonical certificate** (no refinement) => trajectory-
  independent; re-audited standalone with fast_refuse OFF and reproduces
  BYTE-IDENTICAL (converged: the same value re-derived at gen 113 and gen 377).
- The winning genome did exactly what the finding prescribed: it added the
  previously-UNUSED columns k_powers=[2,3,4,5] (C_2^imp-susy) and
  e1_powers=[0,2,4,6,8] (E_1 nulls) plus a full 9-wide x4 tower — a richer
  basis that holds positivity through J=40. Config: f=[2,3,4,6,9,12,16]
  h=[0,3,6,10] e=[0,2,4,6,8] x4=[0..8] x6=[0,5,6,7,8] k=[2,3,4,5]
  e1=[0,2,4,6,8], cap 15000, give 0.05.
- HONEST CAVEAT: at the narrower J<=38 scope the search reached lower c
  (~2.9556), but those are REFINEMENT-TRAJECTORY-DEPENDENT (the fast_refuse
  opt-in changes failing-spin feedback), so the canonical audit of that basis
  gives ~2.9886; not a clean theorem. The ledgered result is the widest-scope,
  trajectory-independent J<=40 = 2.96514.
- Gap to string 2 zeta(3) = 2.4041 narrowed from 0.61 (CMRS) to 0.56.
- METHODOLOGY: this closes the Avenue-A->C arc — a MISDIAGNOSIS (audit wall)
  corrected by direct measurement (min E<0 = basis limit) redirected the
  effort to a richer-basis campaign, which then produced a genuinely new
  certified theorem. Zero spurious certificates across all 462 evals.

## STAGE-2 TAIL-HARDENED BUILD (2026-07-15): exact scope J<=40 -> J<=120

Goal: price the spin truncation and push the exact scope toward all-spin
(the referee-review route). Three build rounds, each caught by the pipeline's
own instruments; no LLM cost (pure local compute).

**RESULT (ledger `g0_extu_tailhard_j120`): g_0 <= 3.114449074 * 8piG/M^6**
(exact rational 616651501245061590189620456603207323780646027859033/
197996976863293703287118043889528424288069801193472), extended-u p_max=9/8,
same axioms as the campaign champion. ITERATION-0 canonical. Exact-rational
per-spin certificates at EVERY even spin J<=120 (certificate audits J<=60
in-loop; standalone processes re-audited 0..60 and extended 62..120 — all
PASS, 12-62s/spin). Stable grid scan positive at every spin to J=240, min
margin +1.72 on the eikonal dip track b~14, growing monotonically to +16.
Functional persisted: artifacts/extu_tailhard_functional.json (39 exact
rationals + config); tail data: artifacts/extu_tailhard_tail.json.

**The truncation tax, measured:** 2.96514 (J<=40) -> 3.1144 (J<=120 hardened).
The sub-CMRS headline was truncation slack, exactly as the monotonicity
argument predicted. If the hardened functional's tail closes (scan evidence,
not proof), the all-spin optimum of this basis lies in [2.96514, 3.1144].

**Two new failure modes found and cured (kept v1/v2 artifacts for the record):**
1. *Traveling eikonal dip* (round 1, c=3.2575, exact J<=60): the binding
   high-spin constraint sits at b=2J/m ~ 14, i.e. m^2 ~ (J/7)^2 — it WALKS
   off any fixed mass grid (hole at J=68-70, m^2~92-100, just past the m^2<=81
   grid). Cure: J^2-scaled shaping/scan grids.
2. *Corner-window spikes* (round 2, c=3.3036, exact only J<=46): for p_max>1
   the p=m kernel corner is INSIDE the smear when m^2 < p_max^2 (=81/64).
   The round-2 functional passed every grid scan (+0.59 margin) yet its exact
   density at J=48, m^2~1.203 is **-1.8e7** (verified in exact rational
   arithmetic at 200 dps): a violent narrow spike invisible to corner-masked
   trapezoid rows AND scans — only the exact audit sees it, and it refused
   correctly (designed asymmetry). EXTENDED-U-SPECIFIC failure mode. Cure:
   dense m^2 in [1, 1.35] rows + the exact audit run INSIDE the refinement
   loop (j_audit=60), which fed corner failures back as refinement points.

**Methodology notes:**
- Exact audits are CHEAP when they pass (12-62s/spin to J=120); only refusals
  grind (334s at a genuine violation). The old "audit resolution wall" is
  fully retired: every wall we ever hit was a genuine violation.
- Fast float64 shaping rows (numpy port of the jet-subtracted stable kernels
  + Gegenbauer three-term recurrence, scratchpad fast_rows.py) validated
  entrywise to ~2e-16 against the production mpmath _stable_row, gated before
  every use; 3762 high-J rows in 4s. Rows normalized per-row (scale-invariant)
  to protect LP conditioning from P_J ~ 1e50 magnitudes.
- certify_g0_upper gained `save_functional=` (persists the exact functional +
  config; previously the pipeline kept the certificate VALUE but discarded the
  FUNCTIONAL, forcing gremlin-prone re-solves to inspect anything).

**Remaining for all-spin:** (E1) continuum fhat(b)>=0 via CMRS App. A.1's 2x2
PSD recast + (E2) finite-(J,m) eikonal-correction bound (Häring–Zhiboedov
semiclassics) — now applied to an EXPLICIT persisted functional already exact
through J=120 with growing margins. Corner window beyond J=120 unprobed except
by audits; scans are blind there.

## STAGE-2 CONCLUSION (2026-07-15/16): margin round + minimal-axiom round —
## the sector's numbers are now 3.0401 (ext-u) / 3.0708 (p_max=1), J<=120

**(E1) reconnaissance on the round-3 functional (3.1144):** f_hat(b) measured
on the continuum (619 pts, b<=400): positive EXCEPT two dips at the 1e-6 level
(-3.4e-6 at b~9.9; also b~16.2) — both BETWEEN the shaping grid's b values —
and a_2 = 0 exactly (asymptotic sign row binding at zero): the tangency tax in
impact-parameter space. Strict all-spin closure FAILS for that functional, but
the obstruction is a 1e-6 margin choice, not structure.

**ROUND 4 (ledger `g0_extu_margin_j120`): g_0 <= 3.040144970704774** — margin-
hardened extended-u certificate (MarginVerifier: strict f_hat >= 1e-3*||row||_1
on b-grid densified over the dips [6,24] step 0.2 + extended to 320; a_2 >=
1e-4; solve() copied from validated parent with margin b_ub only; audit
untouched). ITERATION-0. Exact per-spin at every even J<=120 (in-cert 0..60 +
standalone re-audit + standalone 62..120 extension, all PASS). Scan positive to
J=240 (+1.2 -> +8.5). **f_hat(b) measured POSITIVE at every sampled continuum
point, a_2 = 1e-4 > 0 strictly.** The (E1) obstruction is measured away; also
LOWER than round-3's 3.1144 (both are one-shot certificates, not floors — no
monotonicity paradox). Artifacts: extu_margin_functional/_tail/_fhat.json.

**MINIMAL-AXIOM ROUND (ledger `g0_pm1_tailhard_j120`): g_0 <= 3.070806359835597**
— p_max=1 (NO meromorphy axiom; non-perturbative), arena-1b champion basis with
cap LOWERED 19700 -> 3000 (the big cap outruns the K=26 audit sandwich: J=16-20
small-w refusals with no refinement feedback — first attempt burned; cap-3000
iter-0 passed everything). ITERATION-0, exact per-spin every even J<=120, scan
positive to J=240 (+0.20 min). Truncation tax at minimal axioms: 3.0136 ->
3.0708.

**Physics summary:** exact certification places the D=10 susy g_0 at ~3.01-3.07
across domains/scopes — consistent with and slightly above CMRS's numerical
3.000 (Jmax=42 SDPB), exactly as an exact standard should sit. Extended-u
remains tighter than minimal-axiom AFTER hardening (3.0401 < 3.0708): A2 buys
~0.03. The J<=120 extended-u floor is in [2.96514, 3.0401]; all-spin optimum
>= 2.96514, and <= 3.0401 if the margin functional's tail closes. Remaining for
a genuinely all-spin certificate, sharply delimited: (E1') rigorous enclosures
of f_hat at grid points + explicit modulus of continuity + a_2-dominance (all
certifiable machinery, construction done), and (E2) the finite-(J,m) eikonal-
correction bound (Häring–Zhiboedov semiclassics).

## FLOOR HUNT (2026-07-16, rounds "sweep"+5-8): can plain positivity honestly
## break 3.000? VERDICT: NO in this basis — measured-closed, exactly.

Question: the hardened ext-u floor was bracketed [2.96514, 3.0401]; is the
true certifiable floor below 3.000? Eight measured steps ($0, all local):

1. RAW hardened LP optima (margin shaping, champion basis): 2.731 (p_max=1),
   2.884 (33/32), **2.716 (17/16, new raw sweet spot)**, 2.895 (9/8), 5/4
   infeasible. The J<=40 sweet-spot map (9/8 unique) is OVERTURNED at the raw
   level under hardening — but see (4).
2. **p_max=1 wall = 3.000 = CMRS.** Margin-priced LP lands 3.0007-3.0018 and
   refuses; the reference's number is (numerically) the true optimum of its own
   domain, now validated from the exact side. Cannot be beaten there.
3. give-sweeps + targeted low-J margin blocks + corner-slice margins + all-row
   L1 normalization (fixes a bogus HiGHS 'unbounded' from 30-orders coefficient
   spread) + K-escalation 26->36 in the audit sandwich (SOUND: alternating
   series; passing audits byte-identical; battery 21/21 + regression PASS):
   certifiable stalls at 2.987-3.13 (17/16) with THREE persistent refusals
   (J=2 m~1.4-1.5, J=4 m~1.8-2.0, J=6 small-w); heavy margins certify at
   3.2337 (floor3 artifact; provability tax +0.5; not ledgered, worse than
   record).
4. **EXACT post-mortem (decisive):** the c=2.9870 stall functional's exact
   density BETWEEN grid points: J=2 min E = -0.802 (m~1.37), J=4 -1.63
   (m~1.54), J=6 -1.44, J=8 -28.2 (m~1.07, corner window). Near-optimal
   extended-u functionals develop VIOLENT O(1) dips on sub-grid w-scales
   (~0.01) — refinement chases them, no grid pins them, and there is nothing
   positive for a better audit to prove. The 2.99 stall was NEVER certifiable.

CONCLUSIONS:
- Extended-u plain-positivity certified floor stands at **3.0401** (margin
  functional, exact J<=120); sub-3.000 via plain positivity in this basis is
  MEASURED-CLOSED (violent sub-grid dip structure, not audit technology).
- The dip locations (J=2 m~1.37; J=4 m~1.54; J=6; J=8 m~1.07) are the
  would-be extremal spectrum announcing itself — the cheap extremal-spectrum
  readout now has concrete coordinates.
- A grid+tolerance pipeline would have shipped 2.987 as a bound; its exact
  density is 0.8-28 NEGATIVE at specific states. The designed asymmetry held
  through all 8 rounds: every refusal was correct.
- Remaining levers for sub-3.000: (i) FULL UNITARITY rho_J <= 2 (the
  non-projective upgrade — the one lever that legitimately beats the
  positivity-optimal 3.000; new certificate structure: exact bound on the
  integrated negative part of E via Sturm roots; multi-day build);
  (ii) basis expansion (E_2 nulls / sqrt-2 smears / stable X_8 / LLM campaign
  under the hardened evaluator) — could tighten 3.0401 somewhat, sub-3.000
  doubtful given the generic dip structure near the optimum.

## THREE-TRACK PROGRAM (2026-07-16/17): unitarity / basis expansion / closure

**TRACK A — FULL UNITARITY (rho_J <= 2): CLOSED, double negative.**
(1) Normalization recon assembled the exact penalty formula in our conventions:
gamma_G + gamma_0 g0 >= -(rho_max/pi) Sum_J n_J^{(10)} Int dm^2 m^{-8} [E_J]_-,
n_J^{(10)} = (512 pi^4/3)(J+1)_6(2J+7) — J^7-amplified weights (~2e4*J^7).
(2) U2 RESOLVED AGAINST from the CMRS paper itself (fetched, Sec 2 + 3.8):
the rho<=2 ceiling (their eq 2.13) is defined for COMPONENT-amplitude partial
waves (S=1+iM, rho=Im c_J); the Sec-3.8 susy bracket uses the REDUCED/auxiliary
amplitude's density, for which NO ceiling is stated; CMRS explicitly "make no
use of the upper bound" anywhere. Imposing rho<=2 on our bracket is NOT
licensed; the needed constant = the ceiling the reduced density inherits via
the susy prefactor (m,J-dependent; derivable-but-nontrivial; their pointer for
D=10 full unitarity is GPV 2102.02847).
(3) SCOUT (shaping-level, rho_max=2 as if licensed): the penalty-LP uses ZERO
slack at both p_max=1 and 9/8 (penalty ~2e-15): the J^7 measure makes ANY
negativity more expensive than staying positive. Verdict: even if licensed,
worthless in this formulation. Unitarity is closed as a sub-3.0 lever for the
susy-reduced D=10 pipeline.

**TRACK C — BASIS EXPANSION: E_2 + stable X_8 DELIVERED (production).**
- E_2 null tower: master crossing identity recovered from the CMRS LaTeX
  (K = m^2(2m^2+u)P_J/((m^2-s)(m^2+s+u)), s<->u; EFT side exactly zero);
  E_0 == c0imp-2 and E_1 == e1_expr matched EXACTLY (E_1 thereby
  retro-validated for the first time — it had NO repo validation, gap now
  closed); E_2 = second s-derivative at s=0, same clean audit class as E_1;
  60 checks pass incl. type-II-model heavy-average vanishing with an
  adversarial control (scripts/verify_e2_family.py). Wired as e2_powers
  (absolute end of column order).
- X_8 stable forms: rho(8,r) r=0..4 + factored G(8,r) derived by the same
  Laurent construction that reproduces the k=2,4,6 tables exactly; verified to
  5.7e-51 vs exact _xk_expr(8) (scripts/verify_x8_stable.py, ALL PASS). The
  old "X_8 numerically intractable" verdict is OBSOLETE. Wired into
  _rho/_G/_stable_row + fast rows. Battery 21/21 throughout.
- Round-9 probe (record recipe at 9/8 + X8(0,2,4,6) + E2(0..8) towers vs the
  3.0401 record): running; result appended below.

**TRACK B — ALL-SPIN CLOSURE (E1'): endpoint bug found; route re-priced.**
- BUG (shaping-only, zero theorems affected): _bessel_transform_fn defaults
  pmax=1.0 and every susy caller omitted it — ALL extended-u Bessel shaping
  rows and the earlier fhat continuum measurements used the P=1 transform.
  FIXED at gravity_susy.rows_grid (now passes p_max); fhat re-measured for the
  3.0401 record functional with the TRUE P=9/8 transform: positive except ONE
  dip -1.1e-8 at b~368 (recurring endpoint oscillation, amplitude ~ f(P),
  projected to persist to b~5600). extu_margin_fhat_true.json = authoritative.
- Round-4c (hard f(P)=f'(P)=0): REFUSED at 3.3749 — kills the mid-mass dip
  freedom; too expensive. Route now: SOFT endpoint (|f(P)| <= phi small) +
  raised a_2 floor (dominance horizon b* ~ (A/16a_2)^{2/3} pulled from ~5600
  to a few hundred) = round-4d sweep, targeting whichever functional round-9
  crowns. Rigorous-grid primitive (pure-rational entire-series enclosures of
  t_n(b) with certified truncation bounds) built: scratchpad rigorous_fhat.py.

**TRACK C VERDICT (rounds 9/9b/9c): basis expansion moves the RAW frontier,
not the CERTIFIED one. Record stands at 3.0401.**
- Round 9 (X8+E2 towers, give=0.05): raw LP 2.6329 (vs 2.895 old basis — the
  towers add real LP room) but audit refuses at the usual low-J dips; stalls.
- Round 9b (give=0.16): NEAR MISS — iter-1 at 2.9103 with a SINGLE blocking
  audit (J=0, m~1.0003, the corner-window tip, BETWEEN corner-grid points);
  other iterations refused at corner points (m~1.0004-1.23).
- Round 9c (ultra-fine threshold margin slice m2 in [1.0001,1.015], 25 pts,
  6e-3): CERTIFIED — but at 3.1737 (> record): margins strong enough to tame
  the corner cost more than the towers give back. Artifact kept
  (round9c_basis_functional.json), not ledgered.
- CONCLUSION: the corner-window/near-threshold tangency sets the certified
  extended-u floor ACROSS BASES, recipes, margins, and reserves. All sub-3.0
  levers now measured-closed: truncation slack, margin architecture, p_max
  sweep, unitarity ceiling (unlicensed AND worthless), basis expansion.
  Implication for a hardened-evaluator LLM campaign: expected value LOW for
  the number (the wall is structural); only worth $ for closure-structured
  objectives.

**TRACK B CONCLUSION (rounds 4d A/B/C + deep scan): E1' achieved, E1'-alone
INSUFFICIENT — and the missing ingredient is now measured and named.**
- Round 4d-C (soft endpoint |f(P)|<=0.05 + a_2>=0.01, give=0.11): CERTIFIED
  3.102250152, iteration-0, exact per-spin J<=120 (audits 0..120 all PASS,
  standalone re-audit clean, cG reproduces). TRUE-endpoint f_hat: NON-NEGATIVE
  at all 619 continuum points, f_hat*b^3 rising MONOTONICALLY to the 16a_2 =
  3.51 asymptote (LP pushed a_2 to 0.22, 22x the floor; endpoint oscillation
  crushed as designed). Textbook E1' profile. Artifact:
  extu_soft_C_functional/_fhat/_tail.json. NOT ledgered as record (3.0401
  stands; C's value is the closure experiment).
- BUT: scan negative from J=204 (b~89 track, -0.49 -> -10.9 at 240); deep scan
  to 420: the window MIGRATES along a FIXED-MASS track (m ~ 4.5) with depth
  growing ~linearly in J — the signature of the null columns' explicit
  J(J+7) factors (e1/e2/k kernels) dominating at fixed m. (Caveat: violations
  live in narrow m^2 slivers; coarse grids straddle them — grids at J=204-264
  can read +5 while the sliver at m2=21.0 is -0.5.)
- FINDING: the all-spin problem has TWO closure corners, and the pipeline only
  ever had rows for one: (i) J,m -> inf at fixed b = Bessel/impact-parameter
  closure (imposed since Phase 2); (ii) J -> inf at FIXED m, where
  lim E_J / J^2 is an explicit m-function of the null-column combination
  (P_J-independent PJp1-parts) — analytically derivable, cheap to impose as a
  new row family. The record functional happens to have positive-growing
  fixed-m tails (+16 at J=240); the C functional negative-growing. NEXT
  CONSTRUCTION (scoped, next session): derive the fixed-m closure rows,
  re-solve with BOTH closures + soft endpoint => a candidate that is
  E1'-clean AND fixed-m-tail-positive; then the rigorous grid + explicit
  eikonal remainder (E2) complete the first all-spin gravitational
  certificate.

**EXACT CONFIRMATION of the 4d-C tail diagnosis (reviewer-demanded; the
scan-is-not-proof rule applied to our own scan).** Exact rational arithmetic:
E(J=204, m2~21.2) = -4.13 and E(J=240, m2~22.9) = -15.6 — the violations are
REAL. BUT the "fixed-m J^2-growth / null-column" mechanism read off the deep
scan is REFUTED: at FIXED m2=21.24, E flips from -4.13 (J=204) to +14.9
(J=240). The negative channel MIGRATES ((m2, b) ~ (21.2, 89) -> (22.9, 100))
while deepening — this is the finite-(J,m) eikonal-correction regime, i.e.
precisely (E2), now with exact data points. The proposed "cheap fixed-m
closure rows (lim E/J^2 >= 0)" were designed against a misread and are
WITHDRAWN. Standing lesson formalized: scan-level diagnoses get exact
confirmation BEFORE construction; "analytically derivable, cheap" is
discounted until derived. Paper Outlook corrected accordingly.

**ABLATION (running):** LLM-vs-conventional under the identical evaluator/
seed/budget (462 evals; arms: random hill-climb, GA pop-10; scratchpad/
ablation.py; logs ablation_{random,ga}.jsonl). LLM-arm historical reference:
175/462 certified, best 2.96514@J<=40. Title's "Machine-Guided" is pending
this result.

## 2026-07-17: full referee review applied to the paper (80 findings, 13 blocking)

All 13 blocking + all major findings fixed in docs/paper/main.tex (compiles
clean, 0 undefined refs). The load-bearing corrections:

- **B1 soundness**: "all-spin optimum >= 2.96514" monotonicity claim was
  UNSOUND (2.96514 upper-bounds the unknown true J<=40 optimum; it does not
  equal it). All instances rewritten; no numeric lower bound on the all-spin
  optimum is claimed anywhere.
- **B2/B9**: abstract rewritten (1839 chars, under arXiv 1920); all headline
  numbers presented as certificates for spin-truncated functionals, bounds on
  g0 only conditional on tail positivity. Table 1 recaptioned accordingly.
- **B3/B4**: endpoint bug (P=1 Bessel transform) disclosed in §4.2;
  corrected-transform f-hat facts now in text: 3.1144 clean-but-a2=0; 3.0401
  floor with one -1.1e-8 dip @ b~368; 3.1023 clean+monotone but tail fails
  J~204.
- **B5/B6**: new App. "The wall experiments" (app:wall) — per-route methods,
  numbers table, E1/E2/X8 + master-identity definitions, driver/artifact
  pointers; the 3.0007–3.002 value explicitly scoped as a REFUSED LP stall
  (not a certificate); the 2.93–3.13 interval scoped to that route's sweep.
- **CSDR check RESOLVED (agent read of arXiv:2506.09884 vs 2102.08951 and
  2406.12959)**: Pasiecznik computes g0 M^6/8piG <= 2.97 (her Eq. 32) on the
  smearing region p<=M — equivalent to the STANDARD u in (-M^2,0] domain —
  explicitly matching AKR Eq. (50) = 2.969 (itself a minor improvement on
  CMRS 3.000); her new bounds are D=4 MHV only. NO extended-u g0 exists in
  the literature. Paper hedges replaced with definitive statements; AKR 2.969
  now acknowledged as the sharpest published standard-domain dual bound
  (Setup, Findings, Limitations (v), Related work, Table 1 caption).
- **M-series**: A1 restated on the susy-reduced amplitude; Gegenbauer
  oscillation acknowledged (spectral side NOT manifestly non-negative at
  u<0); the g0 sum-rule identity displayed (improved-C0 tower, gamma_0=-1
  normalization, g0 <= gamma_G * 8piG/M^6); J=42 tolerance finding
  re-attributed to our reconstruction; campaign config named
  (claude-sonnet-5 gen / gemini-3.5-flash judge, T={0.8,1.0}/0.6, 2 islands,
  archive 60); spend corrected $44.6 extu / $76 total; trajectory relabeled
  3.0495(J<=36) -> 3.0400(J<=38) -> 3.0023 -> 2.9928 -> 2.96514(J<=40);
  tab:pmax sweep value 2.9684 disambiguated from campaign 2.96514; D=7 ray
  setup paragraph added; unitarity mentions reconciled (scout = diagnostic,
  no certificate uses rho_J<=2); Outlook de-staled (X8/E2 were built and
  run — basis reach is not the binding constraint).
- **B8**: Reproducibility section now names the concrete layout
  (qgse/verifiers/, results/positivity/{artifacts,drivers}/, grav_rays.jsonl,
  scripts/verify_{x8_stable,e2_family}.py).

Title decision still PENDING ABLATION (running: random arm 27/462 at check,
24 certified, best certified c=3.0051 so far; GA arm queued behind it).

## 2026-07-17: ABLATION COMPLETE — no LLM edge — title -> "Automated Search"

Rebuilt the LLM-vs-conventional ablation after catching two fairness bugs in the
first attempt (v1 operator could not reach the k/C2-susy tower — proven over 20k
mutations, v1 result discarded; and no EWMA per-eval timeout for parity with the
ShinkaEvolve campaign) plus a scope confound (weak 0.6/step fitness discount ->
free search drifts to easy j=36, not comparable to the LLM's j=40 headline).

v2: 12 independent seeds (6 random hill-climb + 6 GA), corrected operator,
EWMA timeout, identical evaluator/seed/budget, pinned to the matched j_audit=40
scope.
- Narrow scope: conventional 2.948 @ j36 matches the LLM's non-promoted narrow
  values. No edge.
- Headline j=40: best-of-12 conventional = 2.96597 (k=[3,4]), gap +0.0008 to the
  LLM's 2.96514 — an effective tie; 2/12 seeds < 3.00, rest plateau ~3.01-3.03.
- Mechanism probe: the C2-susy + E1 towers are jointly beneficial but
  individually infeasible/neutral (a narrow epistatic valley).
VERDICT: no demonstrated LLM search advantage. Title changed to "Exactly
Certified Dispersive Functionals from Automated Search"; new paper Sec. 7
reports the ablation. Data: scratchpad/ablation_v2*, kvalley_probe*,
ablation_findings.md.
