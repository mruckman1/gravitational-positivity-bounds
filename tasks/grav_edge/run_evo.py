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
    ap.add_argument("--max_api_costs", type=float, default=25.0)
    ap.add_argument("--num_islands", type=int, default=2)
    ap.add_argument("--init_program", type=str, default=None)
    args = ap.parse_args()

    if args.offline:
        gen = [f"local/qwen3.5:27b-q8_0@{OLLAMA}"]; jud = [f"local/gemma4:e4b-it-bf16@{OLLAMA}"]; cap = None
    else:
        gen = [os.environ.get("QGSE_GENERATOR_MODEL", "anthropic/claude-sonnet-5")]
        jud = [os.environ.get("QGSE_JUDGE_MODEL", "google/gemini-3.5-flash")]; cap = args.max_api_costs

    rd = args.results_dir or str(REPO / "results" / f"grav_edge_{int(time.time())}")
    print(f"generator={gen} judge={jud} cap={cap} -> {rd}")
    job = LocalJobConfig(eval_program_path=str(HERE / "evaluate.py"),
                         python_executable=str(VENV_PY) if VENV_PY.exists() else sys.executable)
    evo = EvolutionConfig(
        task_sys_msg=(
            "You are designing DISPERSIVE FUNCTIONALS for the D=7 "
            "gravitational positivity HARD EDGE (the max-g3 ray of the "
            "(g_2,g_3) polygon) - real quantum gravity, the largest "
            "improvement headroom in this program. A design = powers "
            "(C_2^imp smearing, [2,16]), x4_powers/x6_powers (crossing "
            "nulls, [0,9], REQUIRED for feasibility, free on the low "
            "side), p_max ('1','9/8','5/4','3/2'; >1 = extended-u domain "
            "(CMRS App B axiom) - NEVER explored for D=7), cap, give, "
            "max_refine [2,6], j_audit {32,36,40} (scope knob: narrower "
            "= easier audit, small fitness discount), delta_base "
            "[1e-5,0.1] (reserve-blend policy; the 35.41 baseline needed "
            "delta=0.064 which dragged the certified slope - better-"
            "conditioned designs need less). Max 30 columns (auto-clipped "
            "beyond). An LP finds the best functional in your design; an "
            "EXACT audit certifies c in g_2 + alpha g_3 + c*8piG >= 0. "
            "Fitness: 0 at our 35.41, 100 at the published 18.0717, more "
            "beyond. Certified-or-nothing: audit failures -50, LP "
            "infeasible -60, no reserve -55 - nothing is gameable. "
            "Design wisdom from the sibling campaign: deep refinement "
            "(max_refine 5+) converges tighter; null towers are the "
            "feasibility currency; the extended domain reorganizes "
            "which constraints bind (expect different power balances "
            "there). Vary one structural element at a time, then "
            "occasionally restructure boldly."
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
