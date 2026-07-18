"""Arena-1b seed: the round-2 champion design (certified c = 3.0136 at
J<=40, baseline domain), now in the ENLARGED design space: extended-u
domain (p_max in {1, 9/8, 5/4, 3/2} — >1 adds the flagged meromorphy
axiom), C_2^imp-susy columns (k_powers, pure-null projected), E_1 nulls
(e1_powers), plus the original towers. NOTE: the champion's powers were
optimized for p_max=1; extended domains reorganize which constraints bind
— bold restructures that exploit the new levers are encouraged."""

# EVOLVE-BLOCK-START
def propose_design():
    """Design space: f_powers [2,16] (C_-2 gravity rule); h_powers [0,10]
    (C_0^imp / g_0 rule); e/x4/x6/e1_powers [0,8] (crossing-null towers,
    free tightness); k_powers [0,6] (C_2^imp-susy, 0 or >=2 entries);
    p_max in {"1","9/8","5/4","3/2"}; cap [50,20000]; give [0.01,0.40];
    max_refine [2,5]. Max 40 columns."""
    return {"f_powers": [2, 3, 4, 6, 9, 12, 16],
            "h_powers": [0, 2, 4, 7, 10],
            "e_powers": [0, 2, 3, 4, 6, 7, 8],
            "x4_powers": [0, 2, 3, 4, 6, 7, 8],
            "x6_powers": [0, 3, 4, 5, 6, 7, 8],
            "k_powers": [],
            "e1_powers": [],
            "p_max": "1",
            "cap": 19700.0, "give": 0.0271, "max_refine": 5}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    return propose_design()
