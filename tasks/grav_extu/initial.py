"""Extended-u richer-basis campaign seed.

Goal: certify g_0 <= c in the EXTENDED-u domain (p_max > 1, meromorphy+
discreteness axiom) at the FULL J<=40 scope, beating the p_max=1 basin floor
of 3.0136 and pushing toward CMRS 3.000 / string 2 zeta(3) = 2.4041.

MEASURED starting fact (this is the whole design problem): the round-2
champion columns at p_max=9/8 CERTIFY 3.0495 only at J<=36 — at J=38/40 that
basis is genuinely NEGATIVE (min E < 0), so the exact audit refuses. This is
a BASIS limit, not an audit limit. The lever is MORE / different columns
(k_powers = C_2^imp-susy, e1_powers = E_1 nulls, richer x4/x6/e towers) that
keep the smeared integrand positive out to J=40 while holding c low.

The seed is that champion basis at p_max=9/8, j_audit=36 (a real certifying
point, c=3.0495). Improve it by GROWING the basis to reach j_audit=40 (the
scope discount rewards it) and/or tightening c below 3.0136."""

# EVOLVE-BLOCK-START
def propose_design():
    """Design space: f_powers [2,16] (C_-2 gravity rule); h_powers [0,10]
    (C_0^imp / g_0 rule); e/x4/x6/e1_powers [0,8] (crossing-null towers,
    free tightness); k_powers [0,6] (C_2^imp-susy, 0 or >=2 entries);
    p_max in {"1","9/8","5/4","3/2"} (>1 = extended-u axiom); j_audit in
    {36,38,40}; cap [50,20000]; give [0.01,0.40]; max_refine [2,5]. Max 40
    columns. To break the J=38/40 violation at p_max>1, spend columns on the
    high-spin null towers (x4/x6/e) and the so-far-unused k_powers/e1_powers."""
    return {"f_powers": [2, 3, 4, 6, 9, 12, 16],
            "h_powers": [0, 2, 4, 7, 10],
            "e_powers": [0, 2, 3, 4, 6, 7, 8],
            "x4_powers": [0, 2, 3, 4, 6, 7, 8],
            "x6_powers": [0, 3, 4, 5, 6, 7, 8],
            "k_powers": [],
            "e1_powers": [],
            "p_max": "9/8",
            "cap": 19700.0, "give": 0.0271, "max_refine": 5,
            "j_audit": 36}
# EVOLVE-BLOCK-END


def run_experiment(**kwargs):
    return propose_design()
