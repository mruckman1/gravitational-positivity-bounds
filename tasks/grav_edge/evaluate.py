"""Arena-2 evaluator: the D=7 HARD EDGE (max-g3 ray). Certified-or-nothing.
Fitness calibration: our 35.41 = 0, published CMRS 18.0717 = 100, room
beyond. Design space: C2imp smearing powers, X4/X6 null smears, extended-u
p_max (axiom-flagged), reserve-blend delta policy, cap/give/max_refine,
j_audit scope knob."""
import os
import argparse
from typing import Any, Dict, List, Optional, Tuple

from shinka.core import run_shinka_eval

BASELINE, PUBLISHED = 35.41, 18.0717
RAY_ALPHA = -1.0 / 3.0 + 3e-4
PMAX_ALLOWED = {"1", "9/8", "5/4", "3/2"}


def _parse(out):
    import sympy as sp
    fp = tuple(sorted(set(int(x) for x in out["powers"])))
    x4 = tuple(sorted(set(int(x) for x in out.get("x4_powers", []))))
    x6 = tuple(sorted(set(int(x) for x in out.get("x6_powers", []))))
    pms = str(out.get("p_max", "1")).strip()
    if pms not in PMAX_ALLOWED:
        raise ValueError(f"p_max must be one of {sorted(PMAX_ALLOWED)}")
    pm = sp.Rational(pms)
    cap = float(out.get("cap", 1000.0))
    give = float(out.get("give", 0.15))
    mr = int(out.get("max_refine", 4))
    ja = int(out.get("j_audit", 40))
    db = float(out.get("delta_base", 1e-3))
    if not fp or min(fp) < 2 or max(fp) > 16:
        raise ValueError("powers must be integers in [2,16], nonempty")
    for t in (x4, x6):
        if t and (min(t) < 0 or max(t) > 9):
            raise ValueError("null powers in [0,9]")
    clipped = False
    while len(fp) + len(x4) + len(x6) > 30:
        if len(x6) >= max(len(x4), 2):
            x6 = x6[:-1]
        elif len(x4) >= 2:
            x4 = x4[:-1]
        else:
            fp = fp[:-1]
        clipped = True
    if not (50.0 <= cap <= 20000.0):
        raise ValueError("cap in [50, 20000]")
    if not (0.01 <= give <= 0.40):
        raise ValueError("give in [0.01, 0.40]")
    if not (2 <= mr <= 6):
        raise ValueError("max_refine in [2,6]")
    if ja not in (32, 36, 40):
        raise ValueError("j_audit in {32, 36, 40}")
    if not (1e-5 <= db <= 0.1):
        raise ValueError("delta_base in [1e-5, 0.1]")
    return fp, x4, x6, pm, cap, give, mr, ja, db, clipped


def validate_design(run_output) -> Tuple[bool, Optional[str]]:
    try:
        _parse(run_output)
    except Exception as e:  # noqa: BLE001
        return False, f"inadmissible design: {e}"
    return True, None


def get_kwargs(run_index: int) -> Dict[str, Any]:
    return {}


def aggregate(results: List[Any]) -> Dict[str, Any]:
    if not results:
        return {"combined_score": -1e9, "public": {"error": "no results"}}
    try:
        fp, x4, x6, pm, cap, give, mr, ja, db, clipped = _parse(results[0])
        from qgse.verifiers.gravity_lp import GravRaySpec, GravityRayVerifier
        import qgse.verifiers.gravity_lp as G
        V = GravityRayVerifier()
        spec = GravRaySpec(ray_alpha=RAY_ALPHA, powers=fp,
                           x_smears=((4, x4), (6, x6)),
                           j_max=40, j_audit=ja, p_max=str(pm),
                           delta_base=db, n_xgrid=300,
                           b_grid=(0.25, 40.0, 200), max_refine=mr)
        # cap/give live inside _solve_lp constants; pass via env-style attrs
        try:
            r = V.certify_ray(spec, log=lambda m: None)
            c = r["c_cert"]
            fit = 100.0 * (BASELINE - c) / (BASELINE - PUBLISHED)
            fit -= 0.6 * ((40 - ja) // 4)
            axiom = " [EXTENDED-u AXIOM]" if pm > 1 else ""
            clipnote = (" NOTE: design clipped to 30 columns."
                        if clipped else "")
            msg = (f"CERTIFIED c = {c:.4f}{axiom} (ours 35.41 = fitness 0, "
                   f"published 18.0717 = 100). Slope certified exactly as "
                   f"{r['ray_alpha_certified']:.6f}." + clipnote)
            status = "certified"
        except RuntimeError as e:
            es = str(e)
            if "infeasible" in es.lower() or "LP failed" in es:
                fit, status = -60.0, "lp_infeasible"
                msg = ("LP INFEASIBLE: this basis cannot satisfy J<=40 "
                       "positivity — add null powers or adjust the basis.")
            elif "reserve" in es.lower():
                fit, status = -55.0, "no_reserve"
                msg = "NO RESERVE with positive margin in this basis."
            else:
                fit, status = -50.0, "audit_failed"
                hint = ("" if pm == 1 else
                        " HINT: at p_max>1 high-spin audits are hard; "
                        "consider j_audit=36 or 32 (scope discount is "
                        "small vs the tightness the extended domain buys).")
                msg = f"AUDIT REFUSED: {es[:120]}" + hint
        return {"combined_score": float(fit),
                "public": {"status": status,
                           "c": (r["c_cert"] if status == "certified"
                                 else None),
                           "p_max": str(pm), "j_audit": ja},
                "private": {}, "text_feedback": msg}
    except Exception as e:  # noqa: BLE001
        return {"combined_score": -1e9, "public": {"status": "error"},
                "private": {"error": str(e)[:300]},
                "text_feedback": f"eval failed: {str(e)[:200]}"}


def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    metrics, correct, err = run_shinka_eval(
        program_path=program_path, results_dir=results_dir,
        experiment_fn_name="run_experiment", num_runs=1,
        get_experiment_kwargs=get_kwargs, validate_fn=validate_design,
        aggregate_metrics_fn=aggregate)
    print("combined_score:", metrics.get("combined_score"),
          "| status:", metrics.get("public", {}).get("status"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--program_path", type=str, default="initial.py")
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()
    main(args.program_path, args.results_dir)
