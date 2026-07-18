"""Arena-2 seed: the certified hard-edge configuration (c = 35.41 at the
D=7 max-g3 ray, J<=40). The published optimum is 18.0717 — a 2x gap, the
largest headroom in the program. New levers vs when 35.41 was set:
extended-u domain (p_max, axiom-flagged), reserve-delta policy, j_audit
scope knob, richer null towers."""

# EVOLVE-BLOCK-START
def propose_design():
    """Design space: powers [2,16] (C_2^imp smearing; the ray's alpha
    columns); x4_powers/x6_powers [0,9] (X_4/X_6 crossing nulls — REQUIRED
    for feasibility, free on the low side); p_max in {"1","9/8","5/4",
    "3/2"} (>1 = extended domain, meromorphy axiom, genuinely unexplored
    for D=7); cap [50,20000]; give [0.01,0.40]; max_refine [2,6];
    j_audit in {32,36,40}; delta_base [1e-5,0.1] (reserve blend policy —
    the 35.41 seed needed delta=0.064 which dragged the ray; SMALLER
    deltas with better-conditioned designs keep the slope). Max 30 cols."""
    return {"powers": [2, 3, 4, 5, 6, 7],
            "x4_powers": [0, 1, 2, 3, 4, 5],
            "x6_powers": [0, 1, 2, 3],
            "p_max": "1",
            "cap": 1000.0, "give": 0.15, "max_refine": 4,
            "j_audit": 40, "delta_base": 1e-3}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    return propose_design()
