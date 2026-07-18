"""SPINNING (spin-resolved) 2D modular-bootstrap verifier — AdS3 quantum gravity
with the full Z(τ, τ̄) and integer spins (Collier–Lin–Yin, arXiv:1608.06241).

The spinless verifier (:mod:`qgse.verifiers.modular`) uses the diagonal β = β̄
reduction — a rigorous relaxation that ignores spin quantization. This module
resolves both chiralities: independent variables (z, z̄) with modular S acting
simultaneously (z, z̄) → (−z, −z̄), primaries labeled (h, h̄) with INTEGER spin
s = h − h̄, and positivity imposed per spin sector. Integer-spin information is
what tightens the bound below the spinless value at matched order.

Formulation (reconciled from CLY 1608.06241 against AHT 1903.06272 / HMR
1905.01319 — the latter two are spinless and provide controls; every
load-bearing identity below was verified by direct computation, see
eval_harness/modular_spin_check.py):

  * Reduced partition function  Ẑ = (b_L b_R)^{1/4} η(τ) η̄(τ̄) Z  — the
    QUARTER-power weight is exactly S-invariant (the half-power is NOT: using
    the spinless kernel per chirality silently yields an invalid bootstrap
    with plausible-looking numbers — the verified trap of this build).
  * Chiral kernel  G(z, x) = e^{z/4} e^{−2π e^z x},  x = h − A_χ,
    A_χ = (c_L − 1)/24 = (c_total − 2)/48. Functional actions are products of
    the chiral polynomials P_m(x) defined by
        (d/dz)^m G(z, x)|_{z=0} = e^{−2πx} P_m(x),
    tied to the validated spinless Q_k by the EXACT diagonal identity
        Q_k(x + x̄) = Σ_m C(k,m) P_m(x) P_{k−m}(x̄).
  * Functional basis: ∂_z^m ∂_z̄^n at z = z̄ = 0 with m + n ODD (total-order
    parity — NOT both-odd), symmetrized (m < n), m + n ≤ Λ.
  * Vacuum: ONE null factor (1 − q) per chirality ⇒ tensor weight pattern
    (1,−1)⊗(1,−1) on (h, h̄) ∈ {0,1}² — the diagonal collapse of which is the
    spinless code's (1,−2,1).
  * Positivity per spin sector s = 0..s_max on Δ ≥ Δ*_s, where the gap kind is
    "dimension" (Δ*_s = max(gap, s)), "scalar" (gap for s=0, s otherwise), or
    "twist" (s + gap).

SOUNDNESS AND SCOPE (worked out rigorously, not assumed): truncating spin at
s_max DROPS constraints on the functional, so a dual-feasible verdict excludes
only spectra all of whose primaries have |spin| ≤ s_max — the certificate says
so explicitly (an s > s_max primary could rescue the spectrum). ALLOWED
verdicts carry no such caveat. Quote bounds only after s_max-stability (the
bound is non-decreasing in s_max). Same 3-way SDPB verdict discipline as every
verifier here: primal feasible → allowed, dual feasible → excluded with the
functional as certificate, anything else → inconclusive error. Exclusions
default to the twist-gapped (no conserved currents) scope unless
allow_currents=True adds the continuous-spin current relaxation.

Cross-verifier control (PROVEN direction): any spinless dual functional embeds
diagonally into the spinning grid via z_{mn} = C(k,m), acts on every state as
e^{−2π(Δ−A)} Q_k(Δ−A) ≥ 0 for all spins including s > s_max, with identical
vacuum normalization — hence spinless-excluded ⇒ spinning-excluded, and
bound_spin(Λ) ≤ bound_spinless(Λ) at matched order (proven for
allow_currents=False; with currents the two verifiers' current blocks start
differently and the embedding argument has not been established).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import sympy as sp
import mpmath as mp

from qgse.interfaces import Candidate, Verifier, VerdictGrounded
from qgse.verifiers.bootstrap import _ensure_env, _in_dir, _REPO
from qgse.verifiers.modular import _poly_eval, _poly_shift

# Chiral polynomials P_m(x): (d/dz)^m [e^{z/4} e^{-2π e^z x}]|_{z=0} = e^{-2πx} P_m(x).
# EXACT sympy coefficients, cached per order. NOTE the 1/4 exponent — do NOT
# substitute the spinless Q_m cache here (kernel e^{s/2} is the diagonal object).
_PPOLY_CACHE: dict = {}


def _p_poly_coeffs(m: int):
    """Exact sympy coefficients [a_0..a_m] of P_m(x) (lowest degree first)."""
    if m not in _PPOLY_CACHE:
        s, x = sp.symbols("s x")
        F = sp.exp(s / sp.Integer(4)) * sp.exp(-2 * sp.pi * sp.exp(s) * x)
        D = sp.diff(F, s, m).subs(s, 0)
        P = sp.expand(D * sp.exp(2 * sp.pi * x))
        poly = sp.Poly(P, x)
        coeffs = [poly.coeff_monomial(x**k) for k in range(poly.degree() + 1)]
        _PPOLY_CACHE[m] = coeffs
    return _PPOLY_CACHE[m]


# -- polynomial helpers (mpf coefficient lists, lowest degree first) ---------- #
def _poly_shift_scaled(coeffs, a, scale):
    """Coefficients in t of P(a + scale*t)."""
    shifted = _poly_shift(coeffs, a)          # P(a + u), coefficients in u
    return [c * (mp.mpf(scale) ** k) for k, c in enumerate(shifted)]


def _poly_mul(c1, c2):
    """Coefficient convolution."""
    out = [mp.mpf(0)] * (len(c1) + len(c2) - 1)
    for i, a in enumerate(c1):
        for j, b in enumerate(c2):
            out[i + j] += a * b
    return out


def _poly_add(c1, c2):
    n = max(len(c1), len(c2))
    return [(c1[i] if i < len(c1) else mp.mpf(0)) +
            (c2[i] if i < len(c2) else mp.mpf(0)) for i in range(n)]


def _component_pairs(lambda_max: int):
    """Unordered functional components {m,n}: 0 <= m < n, m+n odd, m+n <= Λ."""
    pairs = []
    for total in range(1, lambda_max + 1, 2):
        for m in range((total + 1) // 2):
            n = total - m
            if m < n:
                pairs.append((m, n))
    return pairs


@dataclass
class ModularSpinSpec:
    """A gap assumption for the spinning modular bootstrap.

    c_total:   c_L + c_R (= 3ℓ/G_N). Requires c_total > 2 (c_L = c_R > 1).
    gap:       the assumed gap, interpreted per gap_kind.
    gap_kind:  "dimension" — no primary with Δ < gap (Δ*_s = max(gap, s));
               "scalar"    — no SCALAR primary with Δ < gap (other spins at
                             unitarity Δ*_s = s);
               "twist"     — no primary with twist Δ−|s| < gap (Δ*_s = s + gap).
    lambda_max: max total derivative order Λ (odd; components {m<n, m+n odd}).
    s_max:     spin truncation. EXCLUSIONS are scoped to spectra with all
               primaries of |spin| <= s_max; quote bounds only after
               s_max-stability. Default 2Λ (CLY practice: stabilizes ~O(Λ)).
    isolated_points: tuple of (Delta, s) primaries permitted below the gap.
    allow_currents:  adds the continuous-spin current-class relaxation
               (h̄=0 degenerate characters), widening exclusion scope to
               current-ful spectra (spin-truncation scope still applies).
    """

    c_total: float
    gap: float
    gap_kind: str = "dimension"
    lambda_max: int = 11
    s_max: Optional[int] = None
    prec_bits: Optional[int] = None
    dps: Optional[int] = None
    allow_currents: bool = False
    isolated_points: tuple = ()      # ((Delta, s), ...)
    # Optional explicit functional basis: tuple of (m, n) pairs (m < n, m+n odd).
    # None = the full grid up to lambda_max. An explicit SUBSET is a weaker
    # functional space (bound looser/equal) — sound for exclusions; this is the
    # design space for functional-construction search.
    components: Optional[tuple] = None

    def resolved(self):
        s_max = self.s_max if self.s_max is not None else max(20, 2 * self.lambda_max)
        if self.dps is not None:
            dps = self.dps
        else:
            L = self.lambda_max
            big = max(s_max, self.gap, 4)
            dps = max(220, math.ceil(2.2 * L * math.log10(2 * math.pi * big)
                                     + 0.6 * L * math.log10(max(L, 2)) + 80))
        dps = max(dps, 120)     # floor: silent wrong verdicts at absurd user
        prec = self.prec_bits if self.prec_bits is not None else 64 * math.ceil(3.4 * dps / 64)
        prec = max(prec, 448)   # precision are a class-C danger (review finding)
        return s_max, prec, dps

    def delta_star(self, s: int) -> float:
        if self.gap_kind == "dimension":
            return max(self.gap, float(s))
        if self.gap_kind == "scalar":
            return self.gap if s == 0 else float(s)
        if self.gap_kind == "twist":
            return float(s) + self.gap
        raise ValueError(f"unknown gap_kind {self.gap_kind!r}")

    @staticmethod
    def from_dict(d: dict) -> "ModularSpinSpec":
        return ModularSpinSpec(
            c_total=float(d["c_total"]), gap=float(d["gap"]),
            gap_kind=str(d.get("gap_kind", "dimension")),
            lambda_max=int(d.get("lambda_max", 11)),
            s_max=(int(d["s_max"]) if d.get("s_max") is not None else None),
            prec_bits=(int(d["prec_bits"]) if d.get("prec_bits") is not None else None),
            dps=(int(d["dps"]) if d.get("dps") is not None else None),
            allow_currents=bool(d.get("allow_currents", False)),
            isolated_points=tuple((float(p[0]), int(p[1]))
                                  for p in d.get("isolated_points", ())),
            components=(tuple((int(p[0]), int(p[1])) for p in d["components"])
                        if d.get("components") is not None else None),
        )


class ModularSpinVerifier(Verifier):
    domain = "modular_spin"

    def __init__(self, *, work_dir: Optional[str] = None) -> None:
        self.work_dir = Path(work_dir) if work_dir else (_REPO / "results" / "modular_spin_work")

    def known_limits(self) -> list[str]:
        return ["modular_invariance", "unitarity", "virasoro_vacuum_c>1",
                "cL_equals_cR", "integer_spin", "spin_truncation_scope"]

    # -- public entry points ------------------------------------------------ #
    def verify(self, c: Candidate) -> VerdictGrounded:
        try:
            if isinstance(c.meta.get("modular_spin_spec"), dict):
                spec = ModularSpinSpec.from_dict(c.meta["modular_spin_spec"])
            else:
                spec = ModularSpinSpec.from_dict(json.loads(c.artifact))
        except Exception as e:  # noqa: BLE001
            return self._error(f"could not parse modular spin spec: {e!r}")
        return self.verify_spec(spec)

    def verify_spec(self, spec: ModularSpinSpec) -> VerdictGrounded:
        _ensure_env()
        t0 = time.time()
        if spec.c_total <= 2:
            return self._error("c_total must be > 2 (c_L = c_R > 1)")
        if spec.lambda_max < 1 or spec.lambda_max % 2 == 0:
            return self._error("lambda_max must be odd and >= 1")
        if spec.components is not None:
            if not spec.components:
                return self._error("components must be a non-empty pair list")
            for (m, n) in spec.components:
                if not (0 <= m < n and (m + n) % 2 == 1 and m + n <= spec.lambda_max):
                    return self._error(
                        f"component ({m},{n}) invalid: need 0 <= m < n, m+n odd, "
                        f"m+n <= lambda_max={spec.lambda_max}")
            if len(set(spec.components)) != len(spec.components):
                return self._error("duplicate components")
        for (d_i, s_i) in spec.isolated_points:
            if s_i < 0 or d_i <= s_i or d_i <= 0:
                return self._error(
                    f"isolated point (Δ={d_i}, s={s_i}) must satisfy Δ > |s| and "
                    f"Δ > 0 strictly: the Δ = |s| boundary is a DEGENERATE "
                    f"(conserved-current) module whose character this "
                    f"nondegenerate constraint would misrepresent (use "
                    f"allow_currents for the current class)")
        try:
            pmp, n_comp = self._build_pmp(spec)
            run_dir = self.work_dir / self._dirname(spec)
            if run_dir.exists():
                shutil.rmtree(run_dir)
            with _in_dir(run_dir):
                terminate, functional = self._solve(pmp, spec, n_comp)
        except Exception as e:  # noqa: BLE001
            return self._error(f"modular-spin pipeline failed: {e!r}",
                               cost=time.time() - t0)

        cost = time.time() - t0
        if terminate == "found primal feasible solution":
            feasible = True
        elif terminate == "found dual feasible solution":
            feasible = False
        else:
            return self._error(
                f"SDPB inconclusive: terminateReason={terminate!r} — no "
                f"feasibility certificate; raise maxIterations/precision and re-run",
                cost=cost)

        s_max, prec, dps = spec.resolved()
        current_scope = ("current-class positivity imposed (continuous-spin "
                         "relaxation): exclusions cover current-ful spectra"
                         if spec.allow_currents else
                         "twist-gapped scope: no conserved currents besides the "
                         "stress tensor")
        certificate = json.dumps({
            "kind": "sdpb_modular_spin_gap",
            "c_total": spec.c_total, "gap": spec.gap, "gap_kind": spec.gap_kind,
            "lambda_max": spec.lambda_max, "s_max": s_max, "prec_bits": prec,
            "allow_currents": spec.allow_currents,
            "isolated_points": [list(p) for p in spec.isolated_points],
            "assumptions": ("unitary compact 2D CFT, c_L=c_R, modular-invariant "
                            "Z(tau,taubar), INTEGER spins, no primaries below the "
                            f"declared per-spin gaps ({spec.gap_kind} gap {spec.gap:g}) "
                            f"except the declared isolated states; {current_scope}"),
            "scope_spin_truncation": (
                f"exclusion valid for spectra all of whose primaries have "
                f"|spin| <= {s_max}; an s > {s_max} primary is not constrained "
                f"by this functional" if not feasible else None),
            "gravity_reading": ((f"excluded => every consistent AdS3 quantum "
                                 f"gravity with c_total={spec.c_total:g} in scope "
                                 + {"dimension": f"has a primary with Delta <= {spec.gap:g}",
                                    "scalar": f"has a SCALAR primary with Delta <= {spec.gap:g}",
                                    "twist": f"has a primary with twist Delta-|s| <= {spec.gap:g}",
                                    }[spec.gap_kind]
                                 + f"; BTZ threshold = {spec.c_total/24:g}")
                                if not feasible else
                                "allowed at this derivative order"),
            "feasible": feasible, "terminateReason": terminate,
            "dual_functional_y": (functional if not feasible else None),
        }, sort_keys=True)

        verdict = VerdictGrounded(
            valid=feasible,
            fitness=float(spec.gap) if feasible else -1.0,
            certificate=certificate,
            reproduces_known={"modular_invariance": True, "unitarity": True,
                              "integer_spin": True, "gap_allowed": feasible},
            cost=cost,
            details={"feasible": feasible, "terminateReason": terminate,
                     "c_total": spec.c_total, "gap": spec.gap,
                     "gap_kind": spec.gap_kind, "lambda_max": spec.lambda_max,
                     "s_max": s_max},
        )
        verdict.audit()
        return verdict

    def gap_bound(self, spec: ModularSpinSpec, lower: float, upper: float,
                  threshold: float = 0.01) -> dict:
        """Bisect to the max allowed gap. Endpoints verified, inconclusive
        midpoints raise (inherited discipline)."""
        if not (threshold > 0):
            raise ValueError("threshold must be > 0")
        t0 = time.time()
        from dataclasses import replace
        lo, hi = float(lower), float(upper)
        v_lo = self.verify_spec(replace(spec, gap=lo))
        if v_lo.error or not v_lo.valid:
            raise RuntimeError(f"gap_bound precondition failed: lower={lo} not "
                               f"allowed (err={v_lo.error})")
        v_hi = self.verify_spec(replace(spec, gap=hi))
        if v_hi.error or v_hi.valid:
            raise RuntimeError(f"gap_bound precondition failed: upper={hi} not "
                               f"excluded (err={v_hi.error})")
        while hi - lo > threshold:
            mid = 0.5 * (lo + hi)
            v = self.verify_spec(replace(spec, gap=mid))
            if v.error:
                raise RuntimeError(f"bisection failed at gap={mid}: {v.error}")
            lo, hi = (mid, hi) if v.valid else (lo, mid)
        return {"c_total": spec.c_total, "gap_kind": spec.gap_kind,
                "gap_bound_allowed": lo, "gap_bound_excluded_above": hi,
                "lambda_max": spec.lambda_max, "s_max": spec.resolved()[0],
                "cost_s": round(time.time() - t0, 1)}

    # -- PMP construction ---------------------------------------------------- #
    def _dirname(self, spec: ModularSpinSpec) -> str:
        import hashlib
        h = hashlib.sha1(json.dumps(asdict(spec), sort_keys=True,
                                    default=str).encode()).hexdigest()[:10]
        return f"spin_c{spec.c_total:g}_gap{spec.gap:.6f}_{h}"

    def _build_pmp(self, spec: ModularSpinSpec):
        s_max, prec, dps = spec.resolved()
        mp.mp.dps = dps
        A_chi = (mp.mpf(spec.c_total) - 2) / 48
        pairs = (list(spec.components) if spec.components is not None
                 else _component_pairs(spec.lambda_max))
        n_comp = len(pairs)

        # exact -> mpf chiral polynomial coefficients per order
        max_order = spec.lambda_max
        p_mpf = {}
        for m in range(max_order + 1):
            exact = _p_poly_coeffs(m)
            p_mpf[m] = [mp.mpf(str(sp.N(cf, dps + 10))) for cf in exact]

        # vacuum normalization: tensor (1,-1)x(1,-1) at (h, hbar) in {0,1}^2
        norm = []
        for (m, n) in pairs:
            tot = mp.mpf(0)
            for j, wj in ((0, 1), (1, -1)):
                for k, wk in ((0, 1), (1, -1)):
                    xj = mp.mpf(j) - A_chi
                    xk = mp.mpf(k) - A_chi
                    tot += (wj * wk * mp.e ** (-2 * mp.pi * (xj + xk))
                            * (_poly_eval(p_mpf[m], xj) * _poly_eval(p_mpf[n], xk)
                               + _poly_eval(p_mpf[n], xj) * _poly_eval(p_mpf[m], xk)) / 2)
            # NOTE: the symmetrization above double-counts for the vacuum's own
            # symmetric weights; dividing by 2 keeps state/vacuum conventions
            # uniform with the symmetrized state entries below divided the same
            # way. (Uniform positive scaling is irrelevant to feasibility but
            # must be CONSISTENT between norm and blocks.)
            norm.append(tot * 2)

        def s_str(v):
            return mp.nstr(v, dps, strip_zeros=False)

        def continuum_block(x_start, xbar_start):
            """Positivity block: alpha[(Delta*_s + t, s)] >= 0 for t >= 0.
            h-shift per chirality is t/2."""
            polys = []
            for (m, n) in pairs:
                a = _poly_mul(_poly_shift_scaled(p_mpf[m], x_start, mp.mpf("0.5")),
                              _poly_shift_scaled(p_mpf[n], xbar_start, mp.mpf("0.5")))
                b = _poly_mul(_poly_shift_scaled(p_mpf[n], x_start, mp.mpf("0.5")),
                              _poly_shift_scaled(p_mpf[m], xbar_start, mp.mpf("0.5")))
                polys.append(_poly_add(a, b))
            return {
                "DampedRational": {"constant": "1",
                                   "base": s_str(mp.e ** (-2 * mp.pi)), "poles": []},
                "polynomials": [[[ [s_str(c) for c in comp] for comp in polys ]]],
            }

        blocks = []
        for s in range(s_max + 1):
            dstar = mp.mpf(spec.delta_star(s))
            x_start = (dstar + s) / 2 - A_chi
            xbar_start = (dstar - s) / 2 - A_chi
            blocks.append(continuum_block(x_start, xbar_start))

        if spec.components is not None:
            if not spec.components:
                return self._error("components must be a non-empty pair list")
            for (m, n) in spec.components:
                if not (0 <= m < n and (m + n) % 2 == 1 and m + n <= spec.lambda_max):
                    return self._error(
                        f"component ({m},{n}) invalid: need 0 <= m < n, m+n odd, "
                        f"m+n <= lambda_max={spec.lambda_max}")
            if len(set(spec.components)) != len(spec.components):
                return self._error("duplicate components")
        for (d_i, s_i) in spec.isolated_points:
            xi = (mp.mpf(d_i) + s_i) / 2 - A_chi
            xbi = (mp.mpf(d_i) - s_i) / 2 - A_chi
            wi = mp.e ** (-2 * mp.pi * (xi + xbi))
            consts = []
            for (m, n) in pairs:
                consts.append(wi * (_poly_eval(p_mpf[m], xi) * _poly_eval(p_mpf[n], xbi)
                                    + _poly_eval(p_mpf[n], xi) * _poly_eval(p_mpf[m], xbi)))
            blocks.append({
                "DampedRational": {"constant": "1",
                                   "base": s_str(mp.e ** (-2 * mp.pi)), "poles": []},
                "polynomials": [[[ [s_str(c)] for c in consts ]]],
            })

        if spec.allow_currents:
            # current class (h = 1 + t continuum, hbar = 0 degenerate): entries
            # e^{-2πx}P_m(x) ⊗ e^{-2πx̄0}[P_n(x̄0) − e^{−2π}P_n(x̄0+1)] + (m<->n),
            # continuous-j relaxation (rigorous for exclusions).
            xbar0 = -A_chi
            e2pi = mp.e ** (-2 * mp.pi)
            def cur_factor(order):
                return (_poly_eval(p_mpf[order], xbar0)
                        - e2pi * _poly_eval(p_mpf[order], xbar0 + 1))
            x_start = mp.mpf(1) - A_chi
            polys = []
            for (m, n) in pairs:
                a = [c * cur_factor(n) for c in _poly_shift(p_mpf[m], x_start)]
                b = [c * cur_factor(m) for c in _poly_shift(p_mpf[n], x_start)]
                polys.append(_poly_add(a, b))
            blocks.append({
                "DampedRational": {"constant": "1",
                                   "base": s_str(mp.e ** (-2 * mp.pi)), "poles": []},
                "polynomials": [[[ [s_str(c) for c in comp] for comp in polys ]]],
            })

        pmp = {
            "objective": ["0"] * n_comp,
            "normalization": [s_str(v) for v in norm],
            "PositiveMatrixWithPrefactorArray": blocks,
        }
        return pmp, n_comp

    # -- solve ---------------------------------------------------------------- #
    def _solve(self, pmp: dict, spec: ModularSpinSpec, n_comp: int):
        os.environ.setdefault("QGSE_SDPB_IMAGE", "bootstrapcollaboration/sdpb:3.0.0")
        if "wlandry" in os.environ.get("QGSE_SDPB_IMAGE", ""):
            raise RuntimeError("modular-spin verifier requires pmp2sdp; use "
                               "bootstrapcollaboration/sdpb:3.0.0")
        _, prec, _ = spec.resolved()
        Path("pmp.json").write_text(json.dumps(pmp))
        nproc = os.environ.get("QGSE_SDPB_NPROC", "1")
        with open("solver.log", "w") as log:
            subprocess.check_call(["mpirun", "-n", nproc, "pmp2sdp", "-f", "json",
                                   "-i", "pmp.json", "-o", "qgse_mspin",
                                   "-p", str(prec)],
                                  stdout=log, stderr=subprocess.STDOUT)
            subprocess.check_call(["mpirun", "-n", nproc, "sdpb", "-s", "qgse_mspin",
                                   f"--precision={prec}",
                                   "--findPrimalFeasible", "--findDualFeasible"],
                                  stdout=log, stderr=subprocess.STDOUT)
        out = Path("qgse_mspin_out/out.txt")
        terminate = "unknown"
        if out.exists():
            for line in out.read_text().splitlines():
                if "terminateReason" in line:
                    terminate = line.split("=", 1)[1].strip().strip('";')
        functional = None
        ytxt = Path("qgse_mspin_out/y.txt")
        if ytxt.exists():
            # format: "rows cols" header then rows values; with a normalization
            # vector SDPB's y has (n_comp - 1) entries — parse the header, never
            # tail-slice (a tail slice swallows the header token; found by review).
            vals = ytxt.read_text().split()
            try:
                rows = int(vals[0])
                functional = vals[2:2 + rows]
            except Exception:  # noqa: BLE001
                functional = vals
        from qgse.verifiers.bootstrap_mixed import _cleanup_solver_files
        _cleanup_solver_files("qgse_mspin")
        Path("pmp.json").unlink(missing_ok=True)
        return terminate, functional

    def _error(self, msg: str, cost: float = 0.0) -> VerdictGrounded:
        return VerdictGrounded(valid=False, fitness=-1e9, certificate=None,
                               reproduces_known={}, cost=cost, error=msg)
