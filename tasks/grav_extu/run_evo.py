"""Launch the functional-construction experiment: the LLM designs the FUNCTIONAL
BASIS itself (10 derivative components from the 78-pair spin-resolved grid);
SDPB certifies the bound each basis can prove. The first honest LLM test in an
open, non-enumerable design space."""

import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VENV_PY = REPO / ".venv" / "bin" / "python"
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"


def main() -> None:
    load_dotenv(REPO / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default=None)
    ap.add_argument("--num_generations", type=int, default=10000)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max_api_costs", type=float, default=50.0)
    ap.add_argument("--num_islands", type=int, default=2)
    ap.add_argument("--init_program", type=str, default=None)
    args = ap.parse_args()

    if args.offline:
        gen = [f"local/qwen3.5:27b-q8_0@{OLLAMA}"]; jud = [f"local/gemma4:e4b-it-bf16@{OLLAMA}"]; cap = None
    else:
        gen = [os.environ.get("QGSE_GENERATOR_MODEL", "anthropic/claude-sonnet-5")]
        jud = [os.environ.get("QGSE_JUDGE_MODEL", "google/gemini-3.5-flash")]; cap = args.max_api_costs

    rd = args.results_dir or str(REPO / "results" / f"grav_extu_{int(time.time())}")
    print(f"generator={gen} judge={jud} cap={cap} -> {rd}")
    job = LocalJobConfig(eval_program_path=str(HERE / "evaluate.py"),
                         python_executable=str(VENV_PY) if VENV_PY.exists() else sys.executable)
    evo = EvolutionConfig(
        task_sys_msg=(
            "You are designing DISPERSIVE FUNCTIONALS for the D=10 "
            "gravitational R^4 positivity bound - real quantum gravity. A "
            "candidate is a column design: f_powers (p^n smearing the C_-2 "
            "gravity sum rule, n in [2,16]), h_powers (smearing C_0^imp, the "
            "g_0 rule, [0,10]), e_powers/x4_powers (smearing the X_2/X_4 "
            "crossing nulls, [0,8] - FREE tightness, zero low-side cost), "
            "plus cap (coefficient cap) and give (robustness margin). An LP "
            "finds the best functional in your design; an EXACT-arithmetic "
            "audit certifies the resulting ceiling c in g_0 <= c*8piG/M^6, "
            "valid for ALL consistent quantum gravities (spins<=40 scope). "
            "Fitness: 0 at the baseline design (c=4.3307), 68 at the "
            "field's hand-tuned optimum (c=3.000), 100 at the type II "
            "string value (2.4041) - the extremality question: does "
            "consistency pin quantum gravity to the string answer? "
            "Design intuitions: null columns are free constraint-satisfiers "
            "(the baseline uses only 2 X_2 columns - likely far too few); "
            "higher f powers concentrate the smearing at p~1 (small impact "
            "parameter); cap/give trade tightness vs audit margin (big cap "
            "-> knife-edge functionals the audit refuses; audit failures "
            "score 0.5, LP infeasibility 0.3). Every certified point is a "
            "new theorem about quantum gravity - there is nothing to game. "
            "Search systematically: vary one structural element at a time, "
            "use the feedback, occasionally restructure boldly. THIS CAMPAIGN "
            "targets the EXTENDED-u domain (p_max>1, meromorphy+discreteness "
            "axiom): the p_max=1 basin floor is 3.0136 and is already fully "
            "mined, so staying at p_max=1 cannot win. MEASURED FACT: the seed "
            "columns at p_max=9/8 stay positive only to J<=36 (certify "
            "c=3.0495) and genuinely VIOLATE positivity at J=38/40 — a BASIS "
            "limit, not an audit limit. Win by ADDING COLUMNS (the so-far "
            "unused k_powers = C_2^imp-susy and e1_powers = E_1 nulls, plus "
            "richer x4/x6/e towers) that hold positivity out to j_audit=40 "
            "while lowering c below 3.0136, toward 3.000 and 2.4041. The "
            "j_audit scope discount rewards reaching J<=40."
        ),
        init_program_path=(args.init_program or str(HERE / "initial.py")), results_dir=rd,
        num_generations=args.num_generations, language="python",
        llm_models=gen, llm_kwargs={"temperatures": [0.8, 1.0], "max_tokens": 48000, "reasoning_efforts": "max"},
        embedding_model=f"local/{os.environ.get('QGSE_EMBED_MODEL','qwen3-embedding:8b-fp16')}@{OLLAMA}",
        novelty_llm_models=jud, novelty_llm_kwargs={"temperatures": [0.8], "max_tokens": 4000},
        judge_gate_enabled=True, adversarial_judge_models=jud,
        adversarial_judge_kwargs={"temperatures": [0.6], "max_tokens": 4000},
        adversarial_thresholds={"novelty_min": 1.0, "coherence_min": 5.0,
                                "must_be_checkable": True, "must_not_be_disguise": True,
                                "admit_on_parse_error": True},
        use_text_feedback=True, max_api_costs=cap, meta_rec_interval=None)
    db = DatabaseConfig(db_path=str(Path(rd) / "evolution.db"), num_islands=args.num_islands, archive_size=60)
    ShinkaEvolveRunner(evo_config=evo, job_config=job, db_config=db,
                       max_evaluation_jobs=4, max_proposal_jobs=3, max_db_workers=4).run()
    print("Done:", rd)


if __name__ == "__main__":
    main()
