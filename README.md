# Exactly Certified Dispersive Functionals from Automated Search

Reproduction package for the paper **"Exactly Certified Dispersive Functionals
from Automated Search: An Application to Gravitational Positivity Bounds"**
(D=10 maximal-supergravity R⁴ coefficient g₀).

**📄 Paper:** https://mattruckman.com/papers/gravitational-positivity-bounds/

An LLM-guided evolutionary search designs dispersive positivity functionals whose
**sole selection pressure is an exact-arithmetic certificate** — a candidate
counts only if per-spin positivity of its smeared spectral density is *proven* in
exact rational arithmetic, never on a model's sense of plausibility.

> The LLM proposes; an exact certifier disposes.

Floating-point LP/SDP solvers *shape* candidate functionals; a rational-arithmetic
audit (Sturm sequences, Bernstein certificates, alternating-series sandwiches) is
the only thing that certifies. Every bound ships with an explicit scope (spin
range, axioms, domain).

> This repo is scoped to the positivity paper. Four earlier phases of the same
> engine (GR metric, conformal bootstrap, O(N), conjecture engine, AdS3 modular)
> preceded it — see [Project history](#project-history).

---

## Results

**Validation — the scalar EFT-hedron, reproduced with exact *all-spin*
certificates** (`qgse/verifiers/positivity.py`): the analytic single-null optimum
g̃₃ ≥ −10.612487219 to 1e−8 with a complete finite-degree tail proof, the
convergence ladder, and the g̃₄ box [0, ½] exact at both ends.

**D=10 g₀ — exact certificates for spin-truncated dual functionals.** These are
bounds on g₀ *only conditional on tail positivity beyond the audited spins*
(measured on extensive scans; not proven):

| Quantity (units M=1) | this work | scope | ref. |
|---|---|---|---|
| D=10 g₀, pₘₐₓ=1 (reference domain) | 3.0136 | J≤40, +Regge | 3.000 (CMRS) |
| D=10 g₀, pₘₐₓ=1 tail-hardened | 3.0708 | J≤**120** | — |
| D=10 g₀, extended-u tail-hardened | **3.0401** | J≤**120**, +meromorphy | — |
| D=7 hard edge (max-g₃) | 23.1723 | J≤40 | 18.0717 |

An intermediate J≤40 truncation certifies **2.96514** (numerically below the
reference's 3.000), but direct measurement shows this is **truncation slack** — it
does *not* survive tail hardening. Four independent routes to certify below 3.000
all terminate at the same tangency wall near ≈3.0 (the paper's App. D). The
sharpest *published* standard-domain dual bound is **2.969** (Albert–Knop–Rastelli,
2406.12959); the type II string sits at 2ζ(3) ≈ 2.4041.

**Ablation — does the language model help?** A controlled ablation (paper §7)
finds **no demonstrated search advantage** for the LLM proposer: conventional
random-restart and genetic-algorithm searches, given the identical evaluator and
budget, match it at the narrower scope and — as the best of twelve runs —
reproduce the headline value to within 8e−4 at J≤40 (2.96597 vs 2.96514). The
LLM is the search's mutation operator; the contribution is the exact-certified
pipeline around it. Hence the neutral title, "Automated Search."

Full running notebook: [`results/positivity/REPORT.md`](results/positivity/REPORT.md).

---

## Reproducing the paper

Setup (Python 3.11 + editable installs; certificate re-audit needs only
sympy/numpy/scipy):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ./shinka_fork -e .
```

**1. Re-audit the certificates (light; no LLM, no SDPB, no network).** This is the
paper's core claim — that each bound is an exact theorem for its audited spins:

```bash
# rebuild a champion from its artifact and re-audit every even spin bit-for-bit
.venv/bin/python results/positivity/drivers/reaudit_tailhard.py \
    results/positivity/artifacts/extu_margin_functional.json      # -> 3.0401, J<=120
# kernel-identity validation cited in App. D
.venv/bin/python scripts/verify_x8_stable.py
.venv/bin/python scripts/verify_e2_family.py
```

**2. Reproduce the §7 ablation analysis (reads the shipped run logs; no compute):**

```bash
.venv/bin/python results/positivity/ablation/pool_j40.py        # matched-scope verdict
.venv/bin/python results/positivity/ablation/pool_ablation.py   # narrow-scope secondary
```

**3. Re-run a search campaign (heavy; needs an OpenRouter key in `.env`).** Each
eval is a full LP + exact audit (minutes); a full campaign is hours:

```bash
cp .env.example .env      # then edit OPENROUTER_API_KEY=sk-or-...
.venv/bin/python tasks/grav_extu/run_evo.py --num_generations 20 --max_api_costs 20.0
```

Campaigns: `tasks/grav_extu/` (extended-u), `tasks/grav_edge/` (D=7 hard edge),
`tasks/grav_smear3/` (pₘₐₓ=1). Re-running the ablation fleet:
`bash results/positivity/ablation/launch_fleet.sh` (see that folder's README).

**4. Read the paper.** The paper is hosted at
https://mattruckman.com/papers/gravitational-positivity-bounds/. Its LaTeX source
(`docs/paper/main.tex`, a self-contained single-file `article`) is not tracked in
this repo — build it locally with `pdflatex main && pdflatex main` if you have it.

---

## Repository layout

```
qgse/                       # physics verifiers + contracts (kept whole; interdependent import-unit)
  verifiers/gravity_susy.py #   D=10 susy R^4 g_0 certifier — the paper's main instrument
  verifiers/gravity_lp.py   #   D=7 rays: LP shaping + exact continuum audit (Sturm/Bernstein)
  verifiers/gravity_positivity.py # graviton-pole kernels + 21-check assertion battery (CMRS)
  verifiers/positivity.py   #   4D scalar EFT-hedron validation (exact all-spin certificates)
  interfaces.py             #   Candidate / Verifier / Generator / Judge contracts
  (gr.py, bootstrap*.py, modular*.py, conjecture.py, constraints.py remain only
   because the package is an import-unit — they are NOT part of the paper)
tasks/grav_extu/            # D=10 extended-u campaign (2.96514, J<=40)
tasks/grav_edge/            # D=7 hard-edge campaign (23.1723)
tasks/grav_smear3/          # D=10 p_max=1 campaign (3.0136 provenance)
results/positivity/         # the paper's data (git-tracked):
  artifacts/                #   exact-rational certificates + functional configs (JSON)
  drivers/                  #   re-audit + wall-experiment scripts (+ captured outputs)
  ablation/                 #   §7 ablation: corrected harness, valley probe, run logs, pre-registration
  grav_rays.jsonl           #   append-only ledger of every certified value
  ladder.jsonl              #   scalar convergence ladder
  phase2_recon/             #   reference-reconstruction checks (§6 findings)
  REPORT.md                 #   running lab notebook
scripts/                    # verify_x8_stable.py, verify_e2_family.py (App. D wall kernels)
eval_harness/               # positivity_check.py, positivity_ladder.py (scalar validation)
docs/                       # paper LaTeX source (on disk; NOT tracked — paper hosted at mattruckman.com)
shinka_fork/                # vendored ShinkaEvolve fork (search framework; needed to re-run campaigns)
vendor_sdpb_bin/            # sdpb/mpirun shims -> wlandry/sdpb Docker (scalar shaping only)
.env / .env.example         # OpenRouter key + model config (to re-run campaigns)
```

---

## Method

A candidate is a **column structure**: which sum-rule and null families to
include and at which smearing powers, plus the domain endpoint pₘₐₓ and a solver
policy (coefficient cap, robustness margin, refinement depth). Three layers:

1. **Evolutionary search** (`shinka_fork/`) proposes structural mutations; an
   adversarial judge from a different model family pre-filters but never decides
   truth and fails open. Fitness is *certified-or-nothing*.
2. **The exact-arithmetic certifier** (`qgse/verifiers/gravity_susy.py` →
   `gravity_lp.py` → `gravity_positivity.py`): a float LP shapes a functional;
   it is rationalized zero-tolerance; a positivity reserve buys audit margin; then
   a **continuum audit** proves the smeared density E_J(w) ≥ 0 on the full mass
   interval for each audited spin (Sturm / Bernstein). The audit *is* the
   certificate — a `valid` verdict without an auditable exact certificate is a
   hard error.
3. **The assertion battery** (21 checks, all passing) derives every physics kernel
   symbolically and checks it against closed forms at import; the verifier refuses
   to build on failure.

A spurious *exact* certificate is excluded by construction: dropping spins only
*weakens* positivity, rationalization rounds in the safe direction, and tangency
is treated as physics rather than forced positive.

---

## The evolutionary engine (`shinka_fork/`)

`shinka_fork/` is a vendored fork of **ShinkaEvolve** (SakanaAI, Apache-2.0), the
open-source evolutionary program-search framework used as the search loop. It
maintains an *island archive* of candidates, selects parents by fitness+novelty,
asks an LLM to propose mutations, evaluates each in a sandbox, and writes results
back — iterating toward higher fitness. Here the "program" is a dispersive-
functional design, and fitness is the exact certificate. It is vendored (not
pip-installed) because the judge funnel below required source edits and pinning
the tree keeps campaigns reproducible.

It is required to **re-run** the campaigns and the §7 ablation (every
`tasks/*/run_evo.py` and `evaluate.py` imports `shinka`), but **not** to
re-audit certificates — the re-audit path is shinka-free.

### What was changed in the fork

All additive; the master switch defaults off, so stock behavior is preserved.

* **added** `shinka/core/judge_gate.py`, `judge_instrumentation.py`,
  `prompts/prompts_judge.py` — the adversarial-judge funnel.
* **`shinka/core/async_runner.py`** — build the gate in `__init__`; insert it in
  `_generate_evolved_proposal` before eval submission (reusing the terminal-failure
  path so rejection is deadlock-safe); add a `judge_rejected` failure class.
* **`shinka/core/config.py`** — `judge_gate_enabled`, `adversarial_judge_models`,
  `adversarial_judge_kwargs`, `adversarial_thresholds`.
* **`shinka/llm/providers/pricing.csv`** — register `anthropic/claude-sonnet-5`
  and `google/gemini-3.5-flash` (gemini marked mandatory-reasoning).
* **`shinka/llm/providers/openai.py`** — `_extract_output_text()` reads the
  `message` output item of the Responses API, not `output[0]`; without it,
  mandatory-reasoning models return their *reasoning* as content and the judge
  JSON never parses (the funnel would silently fail open on every call).

---

## Design principles

| Principle | How it shows up |
|---|---|
| The grounded verifier is the engine | ~all correctness logic lives in `qgse/verifiers/`; no verifier ever calls an LLM. |
| The judge is a funnel, never an oracle | `shinka_fork/.../judge_gate.py` only *pre-filters*; the exact certificate is the sole source of truth, and the judge fails open. |
| Decorrelate judge from generator | generator and judge are **different model families** (`claude-sonnet-5` vs `gemini-3.5-flash`). |
| Certificates or it didn't happen | a `valid` verdict without an auditable exact certificate raises. |
| State the scope | every bound carries its spin range, axioms, and domain — no silent truncation. |

---

## Project history

Before the positivity program, the same engine + grounded-verifier discipline was
validated across four earlier phases (their code and run outputs are kept in a
local archive, outside this repository):

* **GR (Phase 1).** From a deliberately broken metric ansatz, the full loop
  rediscovered the **Schwarzschild black hole** (symbolically exact) via a
  numeric-first curvature verifier that also handles Kerr in ~1 s.
* **Conformal bootstrap (Phase 3).** SDPB-backed verifier reproduced the **3D
  Ising single-correlator gap bound** (≈1.44) and the **mixed-correlator island**
  (Kos–Poland–Simmons-Duffin), plus the first non-Z2 **O(N)** verifier (XY /
  Heisenberg singlet dimensions), each with dual-functional certificates. Sound
  three-way verdicts (allowed / excluded / inconclusive) after an adversarial
  review caught SDPB stalls being logged as exclusions.
* **Conjecture engine (Phase 4).** A SymPy-exact + mpmath-refutation truth oracle,
  hardened by a review that surfaced and fixed 30 soundness/gaming findings;
  rediscovered Euler's reflection formula offline.
* **AdS3 modular (Phase 4).** A modular-slope assumption-space campaign (SDPB) and
  a spinning-modular functional-design experiment — the setting of an earlier
  measured LLM-vs-baseline design win.

These are accurate as of their runs but are not part of, or needed for, the
positivity paper.

---

## Ceiling

The gravitational certificates are exact for their audited spins, **not all-spin
bounds** — the all-spin dual bound in this framework is open (the tail-closure
program is scoped in the paper's Outlook and App. D). The extended-u results are
**tree-level** (conditional on the meromorphy axiom), not universal swampland
statements. The physics number is a demonstration, pretty modest and above the
reference on the reference's own domain; the contribution is the *rigor* — exact
certificates in place of tolerance-decided numerics — and the tooling that maps
where this functional class stops.

---

## License & citation

This project's code is released under the [MIT License](LICENSE) (© 2026 Matthew
Ruckman). The vendored `shinka_fork/` is a modified fork of
[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) and retains its original
Apache-2.0 license (`shinka_fork/LICENSE`); the fork's modifications are described
under [What was changed in the fork](#what-was-changed-in-the-fork).

If you use this work, please cite the paper:
https://mattruckman.com/papers/gravitational-positivity-bounds/
