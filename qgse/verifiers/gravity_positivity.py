"""Phase-2a foundations: gravitational positivity bounds (CMRS 2102.08951).

Scalar coupled to gravity, general D >= 5: M_low = 8piG[st/u+su/t+tu/s]
- lambda_3^2[1/s+1/t+1/u] - lambda_4 + g_2 sigma_2 + g_3 sigma_3 + ... with
s+t+u=0. Same C_{k,u} residue machinery as the validated scalar verifier
(qgse/verifiers/positivity.py) — CMRS eq (23) is verbatim our B_k rule with
the fixed transfer named u. What gravity changes (recon-verified, artifacts
in results/positivity/phase2_recon/):

  * The graviton hits EXACTLY ONE cell: (k=2, Laurent u^-1) with -8piG,
    piece-split (-1, -1, +1) across st/u, su/t, tu/s. lambda_3^2/lambda_4
    hit NO k>=2 cell (dispersively invisible). All k>=4 Taylor cells are
    G-free: the crossing null tower survives verbatim.
  * The (2,-1) cell is off the Taylor lattice => C_2-family functionals must
    be evaluated at finite u (smeared: f(p) on u=-p^2, p in (0,M]) via the
    IMPROVED kernel C_2^imp (eq 44) whose EFT side is exactly
    8piG/(-u) + 2 g_2 - g_3 u (eq 43) — the contact tower cancels exactly.
  * m -> inf closure: the Bessel/impact-parameter constraint fhat(b) >= 0.
    p-space positivity of f is NEITHER necessary NOR sufficient (D=7 f=p^4
    has fhat < 0 near b ~ 7).

This module ships the general-D kernel layer + the assertion battery (each
recon trap as an executable check, run by eval_harness/gravity_positivity_
check.py BEFORE any solver work builds on it). The LP/smearing/audit stack
(Phase-2a proper: certify the D=7 Table-3 rays) builds on top.

D=4 gravity is REFUSED without an explicit IR-regulator flag: the b-space
graviton measure 8piG b/(D-4) diverges (only b_max-regulated statements
exist, CMRS eq 70).
"""

from __future__ import annotations

import sympy as sp

# symbols: u is the FIXED transfer (CMRS convention), s is dispersed
_S, _U, _M2, _P = sp.symbols("s u m2 p", positive=True)
_D = sp.Symbol("D", positive=True)
_JC2 = sp.Symbol("Jc2", nonnegative=True)     # Casimir J(J+D-3)
G8, L3SQ, L4 = sp.symbols("G8 l3sq l4")       # 8*pi*G, lambda_3^2, lambda_4
_GS = {k: sp.Symbol(f"g{k}") for k in (2, 3, 4, 5, 6, 7)}
_G6P = sp.Symbol("g6p")


# -- general-D partial waves --------------------------------------------------- #
def pj_taylor_D(y, order, D=_D, Jc2=_JC2):
    """P_J(1+y) = sum_r (y/2)^r / (r! ((D-2)/2)_r) prod_{i=0}^{r-1}
    (Jc2 - i(i+D-3)) — the general-D product formula (recon-verified;
    reduces at D=4, Jc2=J(J+1) to positivity._pj_taylor exactly)."""
    out = sp.Integer(0)
    for r in range(order + 1):
        prod = sp.Integer(1)
        for i in range(r):
            prod *= (Jc2 - i * (i + D - 3))
        out += ((y / 2)**r / (sp.factorial(r) * sp.rf((D - 2) / 2, r))) * prod
    return out


def pj_poly(J: int, D, x):
    """Exact Gegenbauer-normalized P_J(x) = 2F1(-J, J+D-3; (D-2)/2; (1-x)/2)
    as a finite sum, P_J(1) = 1 — full polynomial for kernel evaluation at
    finite u."""
    z = (1 - x) / 2
    out = sp.Integer(0)
    for r in range(J + 1):
        out += (sp.rf(-J, r) * sp.rf(J + D - 3, r)
                / (sp.rf(sp.Rational(D - 2, 2) if isinstance(D, int) else
                         (D - 2) / 2, r) * sp.factorial(r))) * z**r
    return sp.expand(out)


def m_low(n_max: int = 7):
    """The gravitational EFT amplitude in (s, u) with t = -s-u eliminated."""
    t = -_S - _U
    sig2 = _S**2 + t**2 + _U**2
    sig3 = _S * t * _U
    M = (G8 * (_S * t / _U + _S * _U / t + t * _U / _S)
         - L3SQ * (1 / _S + 1 / t + 1 / _U) - L4
         + _GS[2] * sig2 + _GS[3] * sig3 + _GS[4] * sig2**2
         + _GS[5] * sig2 * sig3 + _GS[6] * sig2**3 + _G6P * sig3**2
         + _GS[7] * sig2**2 * sig3)
    return M


def residue_rule(expr, k: int):
    """g-side of C_{k,u}: Res_{s=0}[(2s+u)/(s(s+u)) * expr/(s(s+u))^{k/2}].
    Exact via Laurent series in s."""
    integrand = (2 * _S + _U) / (_S * (_S + _U)) * expr / (_S * (_S + _U))**(k // 2)
    # max pole order: k/2 + 2 (from 1/s pieces in expr)
    ser = sp.series(sp.together(integrand) * _S**(k // 2 + 3), _S, 0,
                    k // 2 + 4).removeO()
    return sp.expand(sp.expand(ser).coeff(_S, k // 2 + 2))


# -- the improved kernel (CMRS eq 44) ------------------------------------------ #
def c2imp_kernel(J: int, D, u=_U, m2=_M2):
    """C_2^imp[m^2, J](u): heavy-side action of the improved C_2 sum rule on
    a state (m^2, J). EFT side is exactly 8piG/(-u) + 2 g_2 - g_3 u.
    Regular at u = -m^2 for every even J (recon-verified)."""
    pj = pj_poly(J, D, 1 + 2 * u / m2)
    pj1 = sp.Integer(1)                                  # P_J(1)
    pj1p = sp.Rational(J * (J + D - 3), D - 2) if isinstance(D, int) \
        else sp.nsimplify(J * (J + D - 3)) / (D - 2)     # P'_J(1)
    return ((2 * m2 + u) * pj / (m2 * (m2 + u)**2)
            - (u**2 / m2**3) * ((4 * m2 + 3 * u) * pj1 / (m2 + u)**2
                                + 4 * u * pj1p / (m2**2 - u**2)))


def x2_kernel_forward(D=_D, Jc2=_JC2, m2=_M2):
    """X_{2, u->0}[m^2, J] = (2/(D(D-2))) Jc2 (2 Jc2 - 5D + 4)/m^8 — the
    forward limit of the first u-smeared null (CMRS eq 27; equals the
    validated D=4 n4 row / (4 m^8) at D=4)."""
    return 2 * Jc2 * (2 * Jc2 - 5 * D + 4) / (_D * (_D - 2) * m2**4) \
        if D is _D else 2 * Jc2 * (2 * Jc2 - 5 * D + 4) / (D * (D - 2) * m2**4)


def kappa_D(D):
    """The analytic no-gravity lower-bound coefficient (CMRS eq 35):
    -kappa(D)/M^2 < g_3/g_2. kappa(4) = 10.612... = the CHVD alpha* we
    reproduce to 1e-8 in the scalar verifier — the cross-anchor."""
    Dq = sp.Integer(D) if isinstance(D, int) else D
    return (sp.sqrt((Dq + 3) * (319 * Dq**3 + 76 * Dq**2 - 292 * Dq + 32)
                    / (24 * (Dq - 2)**2 * (Dq + 1) * (Dq + 4)))
            + sp.Rational(6, 12) * (5 * Dq - 2) / (Dq - 2))


# -- assertion battery (each recon trap as an executable check) ----------------- #
def run_assertion_battery(verbose: bool = False) -> list:
    """Run every trap check. Returns list of (name, passed). The solver stack
    must refuse to build if any fails."""
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)

    t = -_S - _U
    # (1) graviton pole split -1,-1,+1 and total -G8/u at k=2; zero at k>=4
    r_st = residue_rule(G8 * _S * t / _U, 2)
    r_su = residue_rule(G8 * _S * _U / t, 2)
    r_tu = residue_rule(G8 * t * _U / _S, 2)
    ck("k=2 graviton piece st/u -> -G8/u", sp.simplify(r_st + G8 / _U) == 0)
    ck("k=2 graviton piece su/t -> -G8/u", sp.simplify(r_su + G8 / _U) == 0)
    ck("k=2 graviton piece tu/s -> +G8/u", sp.simplify(r_tu - G8 / _U) == 0)
    grav = G8 * (_S * t / _U + _S * _U / t + t * _U / _S)
    ck("k=4 graviton -> 0", sp.simplify(residue_rule(grav, 4)) == 0)
    ck("k=6 graviton -> 0", sp.simplify(residue_rule(grav, 6)) == 0)

    # (2) lambda_3^2 / lambda_4 dispersively invisible at k=2,4
    sc = -L3SQ * (1 / _S + 1 / t + 1 / _U) - L4
    ck("lambda3^2/lambda4 -> 0 at k=2", sp.simplify(residue_rule(sc, 2)) == 0)
    ck("lambda3^2/lambda4 -> 0 at k=4", sp.simplify(residue_rule(sc, 4)) == 0)

    # (3) full eq-(39) C_2 low side, term by term
    c2 = sp.expand(residue_rule(m_low(), 2))
    target = (-G8 / _U + 2 * _GS[2] - _GS[3] * _U + 8 * _GS[4] * _U**2
              - 2 * _GS[5] * _U**3 + 24 * _GS[6] * _U**4 - 4 * _GS[7] * _U**5)
    ck("C_2,u == -8piG/u + 2g2 - g3 u + 8g4 u^2 - 2g5 u^3 + 24g6 u^4 - 4g7 u^5",
       sp.simplify(c2 - target) == 0)
    # (4) eq-(40) C_4: G-free, g2/g3-free
    c4 = sp.expand(residue_rule(m_low(), 4))
    ck("C_4,u == 4g4 - 2g5 u + (24g6+g6')u^2 - 8g7 u^3 (G-free)",
       sp.simplify(c4 - (4 * _GS[4] - 2 * _GS[5] * _U
                         + (24 * _GS[6] + _G6P) * _U**2
                         - 8 * _GS[7] * _U**3)) == 0)

    # (5) improved-kernel resummation (eq 44) at J=0: exactly 2/m^4 + 3p^2/m^6
    #     (eq 56) — positive, shape-independent; and eq-45 consistency at J=2.
    k0 = sp.simplify(c2imp_kernel(0, 7, u=-_P**2))
    ck("C_2^imp[J=0](-p^2) == 2/m^4 + 3 p^2/m^6 (eq 56, D-independent)",
       sp.simplify(k0 - (2 / _M2**2 + 3 * _P**2 / _M2**3)) == 0)
    # regularity at u = -m^2 for J=2, D=7 (double pole must cancel)
    k2 = sp.together(c2imp_kernel(2, 7))
    ck("C_2^imp[J=2] regular at u=-m^2 (D=7)",
       sp.simplify(sp.limit(k2 * (_M2 + _U)**2, _U, -_M2)) == 0)

    # (6) X_2 forward limit == eq-27 row; D=4 reduction == n4/(4 m^8)
    from qgse.verifiers.positivity import _ROWS, _X
    x2d4 = x2_kernel_forward(4, _JC2, _M2)
    n4 = _ROWS["n4"][0].subs(_X, _JC2)      # 2X(X-8), X == Jc2 at D=4
    ck("X_2,0(D=4) == n4/(4 m^8)",
       sp.simplify(x2d4 - n4 / (4 * _M2**4)) == 0)

    # (7) general-D P_J product formula: D=4 reduction == legacy _pj_taylor;
    #     and matches exact Gegenbauer at D=7 spot values
    from qgse.verifiers.positivity import _pj_taylor
    y = sp.Symbol("y")
    X4 = sp.Symbol("X")
    legacy = _pj_taylor(y, X4, 4)
    gen = pj_taylor_D(y, 4, 4, X4)          # Jc2 == J(J+1) == X at D=4
    ck("pj_taylor_D(D=4) == legacy _pj_taylor",
       sp.simplify(sp.expand(gen - legacy)) == 0)
    p6 = pj_poly(6, 7, sp.Rational(1, 3))
    p6h = sp.hyper((-6, 6 + 4), (sp.Rational(5, 2),), (1 - sp.Rational(1, 3)) / 2)
    ck("pj_poly(J=6, D=7) == 2F1 value",
       sp.simplify(p6 - sp.hyperexpand(p6h)) == 0)

    # (8) kappa(D) anchors: kappa(4) == CHVD alpha* (our validated 1e-8
    #     number); spot values from the paper
    a4 = sp.Rational(9, 2) + sp.Rational(7, 4) * sp.sqrt(sp.Rational(61, 5))
    ck("kappa(4) == CHVD alpha* = 9/2 + (7/4)sqrt(61/5)",
       sp.simplify(kappa_D(4) - a4) == 0)
    ck("kappa(6) ~ 8.339, kappa(10) ~ 7.218 (paper spot values)",
       abs(float(kappa_D(6)) - 8.339) < 2e-3 and
       abs(float(kappa_D(10)) - 7.218) < 2e-3)

    # (9) single-heavy-scalar end-to-end: M = g^2 sum_ch 1/(m0^2 - ch) has
    #     C_2^imp low side 2g_2 - g_3 u with g_2 = g^2/m0^6, g_3 = 3g^2/m0^8
    #     equal to (g^2/m0^2) * kernel[m0^2, J=0](u)  (recon-verified pair)
    g, m0 = sp.symbols("g m0", positive=True)
    low = 2 * (g**2 / m0**6) - (3 * g**2 / m0**8) * _U
    heavy = (g**2 / m0**2) * c2imp_kernel(0, 7).subs(_M2, m0**2)
    ck("single-scalar model: EFT side == heavy side of C_2^imp exactly",
       sp.simplify(low - heavy) == 0)

    # (11) higher crossing nulls X_8, X_10 (same _xk_kernel machinery, k=8,10):
    #      each must be a genuine null (X_k[J=0] == 0 identically) and lie in
    #      the same pole class the audit handles (regular after (m^2+u)^4, i.e.
    #      pole order < 4, exactly as X_2/4/6). These are the added
    #      free-tightness null directions (Build-1 basis enrichment).
    from qgse.verifiers.gravity_lp import _xk_kernel as _xk_k
    for kk in (8, 10):
        ck(f"X_{kk}[J=0](D=10) == 0 (null)", sp.simplify(_xk_k(kk, 0, 10)) == 0)
        _kJ = sp.together(_xk_k(kk, 2, 10))
        ck(f"X_{kk}[J=2](D=10) pole-order<4 at u=-m^2 (audit class)",
           sp.simplify(sp.limit(_kJ * (_M2 + _U)**4, _U, -_M2)) == 0)

    # (10) D=4-refusal guard exists at the spec layer (checked in harness)
    return checks
