"""Arena-1b evaluator: certified-or-nothing on the ENLARGED instrument —
extended-u domain (meromorphy axiom, flagged), E1 + C2imp-susy towers,
large-J viability rows. Fitness calibration unchanged: baseline 4.3307 = 0,
CMRS 3.000 = 68.4, string 2.4041 = 100."""
import os
import argparse
from typing import Any, Dict, List, Optional, Tuple

from shinka.core import run_shinka_eval

BASELINE, STRING = 4.3307, 2.404113806
PMAX_ALLOWED = {"1": "1", "9/8": "9/8", "5/4": "5/4", "3/2": "3/2"}


def _parse(out):
    import sympy as sp
    fp = tuple(sorted(set(int(x) for x in out["f_powers"])))
    hp = tuple(sorted(set(int(x) for x in out["h_powers"])))
    ep = tuple(sorted(set(int(x) for x in out.get("e_powers", []))))
    xp = tuple(sorted(set(int(x) for x in out.get("x4_powers", []))))
    x6p = tuple(sorted(set(int(x) for x in out.get("x6_powers", []))))
    kp = tuple(sorted(set(int(x) for x in out.get("k_powers", []))))
    e1p = tuple(sorted(set(int(x) for x in out.get("e1_powers", []))))
    pms = str(out.get("p_max", "1")).strip()
    if pms not in PMAX_ALLOWED:
        raise ValueError(f"p_max must be one of {list(PMAX_ALLOWED)}")
    pm = sp.Rational(pms)
    cap = float(out.get("cap", 3000.0))
    give = float(out.get("give", 0.10))
    mr = int(out.get("max_refine", 3))
    ja = int(out.get("j_audit", 40))
    if ja not in (36, 38, 40):
        raise ValueError("j_audit must be 36, 38, or 40")
    if not fp or min(fp) < 2 or max(fp) > 16:
        raise ValueError("f_powers in [2,16], nonempty")
    if not hp or min(hp) < 0 or max(hp) > 10:
        raise ValueError("h_powers in [0,10], nonempty")
    for t in (ep, xp, x6p, e1p):
        if t and (min(t) < 0 or max(t) > 8):
            raise ValueError("null powers in [0,8]")
    if kp and (min(kp) < 0 or max(kp) > 6):
        raise ValueError("k_powers in [0,6]")
    if len(kp) not in (0, 2, 3, 4, 5, 6, 7):
        raise ValueError("k_powers needs 0 or >=2 entries (exact projection)")
    ncols = sum(map(len, (fp, hp, ep, xp, x6p, kp, e1p)))
    clipped = False
    # REPAIR OPERATOR: oversized designs are clipped (drop highest powers
    # from the largest null towers first), evaluated, and told about it —
    # a wasted -1e9 eval becomes a valid data point + teaching signal.
    while sum(map(len, (fp, hp, ep, xp, x6p, kp, e1p))) > 40:
        towers = sorted((("e1p", e1p), ("x6p", x6p), ("x4p", xp),
                         ("ep", ep), ("kp", kp), ("hp", hp)),
                        key=lambda t: -len(t[1]))
        name, tow = towers[0]
        if len(tow) <= (2 if name == "kp" else 1):
            fp = fp[:-1]            # last resort: trim f tail
        else:
            tow = tow[:-1]
        if name == "e1p": e1p = tow
        elif name == "x6p": x6p = tow
        elif name == "x4p": xp = tow
        elif name == "ep": ep = tow
        elif name == "kp": kp = tow
        elif name == "hp": hp = tow
        clipped = True
    if not (50.0 <= cap <= 20000.0):
        raise ValueError("cap in [50, 20000]")
    if not (0.01 <= give <= 0.40):
        raise ValueError("give in [0.01, 0.40]")
    if not (2 <= mr <= 5):
        raise ValueError("max_refine in [2,5]")
    return (fp, hp, ep, xp, x6p, kp, e1p, pm, cap, give, mr,
            clipped, ja)


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
        (fp, hp, ep, xp, x6p, kp, e1p, pm, cap, give, mr,
         clipped, ja) = _parse(results[0])
        from qgse.verifiers.gravity_susy import (SusyR4Verifier,
                                                 certify_g0_upper)
        V = SusyR4Verifier(f_powers=fp, h_powers=hp, e_powers=ep,
                           x4_powers=xp, x6_powers=x6p, k_powers=kp,
                           e1_powers=e1p, p_max=pm)
        V.fast_refuse = True   # campaign: refuse-only fast path on clearly-
        # negative spins (sound; never blocks a certifiable design). Extended-u
        # J=40 designs violate often; this keeps violating evals ~seconds not
        # ~45min. Certified results still go through the full exact audit.
        try:
            r = certify_g0_upper(j_max=40, j_audit=ja, max_refine=mr,
                                 log=lambda m: None, verifier=V,
                                 cap=cap, give=give)
            c = r["c_cert"]
            fit = 100.0 * (BASELINE - c) / (BASELINE - STRING)
            # scope discount: narrower audit scope = weaker theorem class;
            # 0.6 fitness per dropped spin-step keeps wide scope preferred
            fit -= 0.6 * ((40 - ja) // 2)
            axiom = " [EXTENDED-u AXIOM]" if pm > 1 else ""
            clipnote = (" NOTE: your design exceeded 40 columns and was "
                        "CLIPPED (highest powers dropped from largest "
                        "towers) before evaluation — stay within 40 to "
                        "control your own design." if clipped else "")
            msg = (f"CERTIFIED g_0 <= {c:.4f} * 8piG/M^6{axiom} "
                   f"(baseline 4.3307, CMRS 3.000, string 2.4041)."
                   + clipnote)
            status = "certified"
        except RuntimeError as e:
            es = str(e)
            if "infeasible" in es.lower():
                fit, status = 0.3, "lp_infeasible"
                msg = ("LP INFEASIBLE at J<=40 + large-J viability rows: "
                       "add null columns or relax powers.")
            else:
                fit, status = 0.5, "audit_failed"
                hint = ("" if pm == 1 else
                        " NOTE (measured): at p_max>1 the champion-columns "
                        "basis VIOLATES positivity at J=38/40 (min E<0) — a "
                        "BASIS limit, not an audit limit. To certify J<=40 in "
                        "the extended domain, ADD COLUMNS (k_powers, e1_powers, "
                        "more x4/x6/e powers) so the functional stays positive "
                        "at high spin. Champion columns at j_audit=36 certify "
                        "3.0495; a richer basis may reach J<=40 and beat the "
                        "p_max=1 floor 3.0136.")
                msg = f"AUDIT REFUSED: {es[:130]}" + hint
        return {"combined_score": float(fit),
                "public": {"status": status,
                           "c": (r["c_cert"] if status == "certified"
                                 else None),
                           "p_max": str(pm)},
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
