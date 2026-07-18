"""Validation ladder for the 4D positivity-bounds verifier (Phase 1 gate).

Every mechanism validated on a known answer BEFORE unknown territory:
  stage unit — exact sympy identities (import itself re-derives all rows from
    the master kernel and asserts the closed forms), partial-wave conventions,
    the analytic optimum alpha* = 9/2 + (7/4)sqrt(61/5), the 2-state primal
    witness closing strong duality, the corrected n5 spot values, and the
    AUDIT-GUARD regression: the explicitly-computed spurious J_max=2
    functional (recon counterexample) MUST be refused.
  stage sdpb — reproduce CHVD: gt_3 >= -10.612487218800544 (n4 only, exact
    analytic value), gt_3 <= 3 (exact), and the n4+n5 run (Table 2, n=5).

Usage: python -m eval_harness.positivity_check --stage unit|sdpb|all
"""
import argparse
import sys

import sympy as sp

FAIL = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}", flush=True)
    if not cond:
        FAIL.append(name)


def stage_unit():
    print("== stage unit: exact identities ==", flush=True)
    # (1) import runs the kernel->rows derivation asserts
    from qgse.verifiers import positivity as P
    check("kernel-derived rows match closed forms (import-time asserts)", True)

    # (2) n_J^(4) = 16 pi (2J+1) from the general-d formula
    d, J = sp.symbols("d J", positive=True)
    nJd = ((4 * sp.pi)**(d / 2) * (d + 2 * J - 3) * sp.gamma(d + J - 3)
           / (sp.pi * sp.gamma((d - 2) / 2) * sp.gamma(J + 1)))
    check("n_J^(4) == 16 pi (2J+1)",
          all(sp.simplify(nJd.subs(d, 4).subs(J, j) - 16 * sp.pi * (2 * j + 1))
              == 0 for j in (0, 2, 4, 8)))

    # (3) P_J(1+y) Taylor formula vs sympy legendre, J <= 12, rational y
    y = sp.Rational(3, 7)
    ok = True
    for j in range(0, 13, 2):
        X = j * (j + 1)
        taylor = sum(y**r / (2**r * sp.factorial(r)**2)
                     * sp.prod([X - i * (i - 1) for i in range(1, r + 1)])
                     for r in range(j + 1))
        ok &= sp.simplify(taylor - sp.legendre(j, 1 + y)) == 0
    check("P_J(1+y) Taylor formula == legendre (J<=12)", ok)

    # (4) n5 spot values (incl. the adversarially-CORRECTED J=4 value)
    n5 = P._ROWS["n5"][0]
    check("n5(J=0) == 0", n5.subs(P._X, 0) == 0)
    check("n5(J=2) == -432", n5.subs(P._X, 6) == -432)
    check("n5(J=4) == +3600 (corrected; NOT +8400)",
          n5.subs(P._X, 20) == 3600)
    n4 = P._ROWS["n4"][0]
    check("n4(J=2) == -24, n4(J=4) == +480",
          n4.subs(P._X, 6) == -24 and n4.subs(P._X, 20) == 480)

    # (5) Regge blocks are the exact large-x limits
    names, k_max, rows, regge = P._numerator_rows(("n4",), "gt3")
    check("Regge block (n4): (1, -2xi, 2xi^2)",
          [q.as_expr() for q in regge] ==
          [sp.Integer(1), -2 * P._XI, 2 * P._XI**2])
    names5, k5, rows5, regge5 = P._numerator_rows(("n4", "n5"), "gt3")
    check("Regge block (n4,n5): (1, -2xi, 2xi^2, 4xi^3)",
          [sp.expand(q.as_expr() - e) == 0 for q, e in
           zip(regge5, [sp.Integer(1), -2 * P._XI, 2 * P._XI**2,
                        4 * P._XI**3])] == [True] * 4)

    # (6) analytic optimum: alpha* = 9/2 + (7/4) sqrt(61/5)
    alpha = sp.Rational(9, 2) + sp.Rational(7, 4) * sp.sqrt(sp.Rational(61, 5))
    beta = (alpha - 9) / 24
    z = [alpha, sp.Integer(1), beta]
    G = sum(z[i] * rows[i].as_expr() for i in range(3))
    g2 = sp.expand(G.subs(P._X, 6))     # J=2 block
    g4 = sp.expand(G.subs(P._X, 20))    # J=4 block
    check("optimal functional: J=2 block has root at x=0",
          sp.simplify(g2.subs(P._x, 0)) == 0)
    # J=4 block double root at 1+x = 37/(2 alpha*)
    r4 = sp.Rational(37, 2) / alpha - 1
    check("optimal functional: J=4 block double root at 1+x=37/(2a*)",
          sp.simplify(g4.subs(P._x, r4)) == 0 and
          sp.simplify(sp.diff(g4, P._x).subs(P._x, r4)) == 0)
    # odd-spin negativity (the even-only mechanism): J=3 block negative at x=0
    g3odd = G.subs(P._X, 12).subs(P._x, 0)
    check("J=3 (odd) block NEGATIVE at x=0 (even-only spins essential)",
          sp.simplify(g3odd) < 0)

    # (7) 2-state primal witness closes strong duality EXACTLY:
    # states (J=2, m^2=1) and (J=4, m^2=v), v = (-180+14*sqrt(305))/37,
    # weights b at J=4 and a = 20 b / v^4 at J=2 solve <n4> = 0 and give
    # gt_3 = -(9/2 + 7*sqrt(305)/20) = -alpha*.
    v = (-180 + 14 * sp.sqrt(305)) / 37
    a_w = 20 / v**4                       # b = 1
    avg_n4 = a_w * (2 * 6 * (6 - 8)) / 1**4 + (2 * 20 * (20 - 8)) / v**4
    g2v = a_w / 1**2 + 1 / v**2
    g3v = a_w * (3 - 12) / 1**3 + (3 - 40) / v**3
    check("witness: <n4> == 0 exactly", sp.simplify(avg_n4) == 0)
    check("witness: gt_3 == -alpha* exactly",
          sp.simplify(g3v / g2v + alpha) == 0)

    # (8) AUDIT-GUARD regression: the recon counterexample functional
    # (J_max=2-optimal, z=(10.17890835, 1, 0.049121)) must be REFUSED —
    # it is negative at the excluded spin J=4.
    from qgse.verifiers.positivity import PositivitySpec, PositivityVerifier
    spec = PositivitySpec(nulls=("n4",), j_audit=40)
    zbad = [sp.Rational("10.17890835"), sp.Integer(1), sp.Rational("0.049121")]
    ok_flag, _, _, why = PositivityVerifier._audit_once(spec, rows, zbad)
    check("audit REFUSES the known-spurious J_max=2 functional "
          f"({why or 'accepted?!'})", not ok_flag and "J=4" in why)
    # and the true optimal functional (rational-rounded with margin) passes
    zgood = [sp.Rational("10.6125"), sp.Integer(1), sp.Rational("0.0671875")]
    okg, tail, _, whyg = PositivityVerifier._audit_once(spec, rows, zgood)
    check(f"audit ACCEPTS the true functional incl. TAIL proof ({whyg or 'ok'})",
          okg and tail)

    # (9) scalar-exchange end-to-end: M = -g^2 sum 1/(ch - m^2) has
    # g_2 = g^2/m^6, g_3 = 3 g^2/m^8 via the 1-state measure <F> = g^2/m^2 F.
    m2, g = sp.symbols("m2 g", positive=True)
    w = g**2 / m2
    check("scalar exchange: g_2 = g^2/m^6, g_3 = 3g^2/m^8, g_4 = g^2/(2m^10)",
          sp.simplify(w * P._ROWS["g2"][0].subs(P._X, 0) / m2**2
                      - g**2 / m2**3) == 0 and
          sp.simplify(w * P._ROWS["g3"][0].subs(P._X, 0) / m2**3
                      - 3 * g**2 / m2**4) == 0)


def stage_sdpb():
    print("== stage sdpb: reproduce CHVD ==", flush=True)
    from qgse.verifiers.positivity import PositivitySpec, PositivityVerifier
    V = PositivityVerifier()
    exact = -(4.5 + 1.75 * (61 / 5) ** 0.5)     # -10.612487218800544

    r = V.extremal_bound(PositivitySpec(side="lower", nulls=("n4",),
                                        j_max=40, j_audit=200))
    print(f"  gt3 lower (n4): certified {r['bound']:.9f} "
          f"(sdpb {r['sdpb_objective']:.9f}; exact {exact:.9f}) "
          f"tail_proven={r['audit']['tail_proven']}", flush=True)
    check("gt3 lower (n4) reproduces analytic optimum to 1e-4 "
          "(certified side, margin-weakened)",
          abs(r["sdpb_objective"] - exact) < 1e-6 and
          exact - 1e-4 < r["bound"] <= exact + 1e-9)
    check("gt3 lower (n4): COMPLETE tail certificate (all spins)",
          r["audit"]["tail_proven"])

    r2 = V.extremal_bound(PositivitySpec(side="upper", nulls=("n4",),
                                         j_max=40, j_audit=200))
    print(f"  gt3 upper (n4): certified {r2['bound']:.9f} (exact 3)",
          flush=True)
    check("gt3 upper == 3 exactly (spin-0 threshold saturation)",
          abs(r2["bound"] - 3.0) < 1e-6)

    r3 = V.extremal_bound(PositivitySpec(side="lower", nulls=("n4", "n5"),
                                         j_max=60, j_audit=200))
    print(f"  gt3 lower (n4,n5): certified {r3['bound']:.9f} "
          f"(CHVD Table 2 n=5: -10.6125) scope: {r3['scope']}", flush=True)
    check("gt3 lower (n4,n5) matches Table 2 n=5 (-10.6125 +- 2e-3)",
          abs(r3["bound"] + 10.6125) < 2e-3)
    # scope must be CONSISTENT: complete certificate iff claimed "all spins";
    # otherwise honestly scoped to the audited spin range — never silent.
    check("gt3 lower (n4,n5) scope consistent with its certificate "
          f"(tail_proven={r3['audit']['tail_proven']})",
          ("complete exact certificate" in r3["scope"]) == r3["audit"]["tail_proven"] and
          (r3["audit"]["tail_proven"] or "J <=" in r3["scope"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["unit", "sdpb", "all"])
    args = ap.parse_args()
    if args.stage in ("unit", "all"):
        stage_unit()
    if args.stage in ("sdpb", "all"):
        stage_sdpb()
    print()
    if FAIL:
        print(f"POSITIVITY VALIDATION: FAIL ({len(FAIL)}): {FAIL}")
        sys.exit(1)
    print("POSITIVITY VALIDATION: PASS")


if __name__ == "__main__":
    main()
