"""2D modular-bootstrap verifier — direct quantum-gravity discrimination.

The most gravity-pointed verifier in the suite. A unitary, compact 2D CFT with
central charges c_L = c_R defines a quantum gravity theory in AdS3 (Brown–
Henneaux: c_total = c_L + c_R = 3ℓ/G_N). Bounding the lightest primary Δ1 at
given c_total is therefore a statement of the form:

    "ANY consistent quantum gravity in AdS3 must contain a state with
     Δ ≤ f(c_total)".  Landmarks (in c_total = c_L + c_R convention):
     BTZ black holes start at Δ = c_total/24; Hellerman's classic bound is
     Δ1 ≤ c_total/12 + 0.474 (twice BTZ); the best known numerical slope
     (Afkhami-Jeddi–Hartman–Tajdini) is ≈ c_total/18.2 at very high order.
     Bounds from this verifier land between BTZ and Hellerman and tighten
     toward the AHT value as n_derivs grows.

Method (spinless modular bootstrap, Hellerman 2009 / Friedan–Keller lineage):
Z(β) decomposed into Virasoro characters (c > 1: vacuum + nondegenerate
primaries), reduced so modular S becomes β -> 4π²/β symmetric:

    g(β) = √β [ e^{β A}(1-e^{-β})²  +  Σ_prim e^{-β(Δ - A)} ],
    A := (c_total - 2)/24,   g(β) = g(4π²/β).

With s = ln(β/2π), g is even in s, so odd s-derivatives at s=0 annihilate it.
A linear functional α = Σ_k z_k (d/ds)^{2k+1}|_{s=0} with α[vacuum] = 1 and
α[Δ] ≥ 0 for all Δ ≥ Δ*  is a machine-checkable PROOF that no unitary spectrum
with gap Δ* exists. Each α[Δ] = e^{-2πx} Q(x) with x = Δ - A and Q polynomial,
so functional-existence is exactly SDPB's polynomial-matrix-program feasibility
— the same solver, verdict semantics, and certificate discipline as the other
bootstrap verifiers (primal feasible -> gap ALLOWED; dual feasible -> EXCLUDED
with the dual functional as certificate; anything else -> inconclusive error).

Scope/honesty (per adversarial review): the spinless reduction ignores spin
quantization (a rigorous relaxation), BUT the default positivity cone contains
only NONDEGENERATE characters. Conserved spin-J currents (h̄=0) have degenerate
characters — signed combinations χ̂(Δ)−χ̂(Δ+1) — on which a default functional
can go negative. Therefore default exclusions are rigorous for **twist-gapped
(Virasoro-only) theories: no conserved currents besides the stress tensor** —
the same convention under which the Hellerman/HMR benchmarks are quoted; they
do NOT rule out current-ful duals (e.g. higher-spin AdS3 gravities). Setting
``allow_currents=True`` adds the current-class positivity constraint
e^{-2πx}[Q(x) - e^{-2π}Q(x+1)] ≥ 0 (continuous-spin relaxation), making
exclusions valid for ALL unitary compact c_L=c_R CFTs at slightly weaker bounds.
Exact benchmarks (Hartman–Mazáč–Rastelli sphere packing): the optimal bound is
Δ1 = 1 at c_total = 8 and Δ1 = 2 at c_total = 24; finite order approaches these
from above. No PyCFTBoot needed: characters give polynomials directly; the PMP
JSON is written by this module and solved via the Docker SDPB shims (requires
an image with pmp2sdp, i.e. bootstrapcollaboration/sdpb — multi-arch).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import sympy as sp
import mpmath as mp

from qgse.interfaces import Candidate, Verifier, VerdictGrounded
from qgse.verifiers.bootstrap import _ensure_env, _in_dir, _REPO

# Exact functional-basis polynomials, cached per derivative order:
#   (d/ds)^m [ e^{s/2} exp(-2π e^s x) ] |_{s=0}  =  e^{-2πx} Q_m(x)
# Q_m has degree m with exact (rational × π-power) coefficients; c enters only
# through evaluation points, so this cache is global.
_QPOLY_CACHE: dict = {}


def _q_poly_coeffs(m: int):
    """Exact sympy coefficients [a_0..a_m] of Q_m(x) (lowest degree first)."""
    if m not in _QPOLY_CACHE:
        s, x = sp.symbols("s x")
        F = sp.exp(s / 2) * sp.exp(-2 * sp.pi * sp.exp(s) * x)
        D = sp.diff(F, s, m).subs(s, 0)
        Q = sp.expand(D * sp.exp(2 * sp.pi * x))
        poly = sp.Poly(Q, x)
        coeffs = [poly.coeff_monomial(x**k) for k in range(poly.degree() + 1)]
        _QPOLY_CACHE[m] = coeffs
    return _QPOLY_CACHE[m]


@dataclass
class ModularSpec:
    """A gap assumption for AdS3 quantum gravity / 2D CFT.

    c_total:  c_L + c_R (= 3ℓ/G_N by Brown–Henneaux). Requires c_total > 2.
    gap:      assumed lowest nonvacuum primary dimension Δ*.
    n_derivs: number N of odd-derivative functionals (orders 1,3,..,2N-1).
              Higher N = tighter bound, slower solve. N >= 2 for finite bounds.
    prec_bits / dps: SDPB precision and decimal digits for coefficients.
    """

    c_total: float
    gap: float
    n_derivs: int = 12
    prec_bits: int = 768
    dps: int = 220
    # False (default): twist-gapped convention (no conserved currents) — matches
    # the Hellerman/HMR benchmarks. True: adds current-class positivity so
    # exclusions cover current-ful spectra too (slightly weaker bounds).
    allow_currents: bool = False
    # Isolated primaries ALLOWED below the gap: spectrum assumption becomes
    # "vacuum + these states + arbitrary primaries above gap". Each adds a
    # single-point positivity constraint on the functional (α[Δ_i] ≥ 0), so an
    # exclusion certifies: NO unitary spectrum whose only sub-gap primaries are
    # exactly these isolated states exists. Enables two-gap / spectral-
    # determinacy questions (e.g. one state at the BTZ threshold + desert).
    isolated_points: tuple = ()

    @staticmethod
    def from_dict(d: dict) -> "ModularSpec":
        return ModularSpec(
            c_total=float(d["c_total"]), gap=float(d["gap"]),
            n_derivs=int(d.get("n_derivs", 12)),
            prec_bits=int(d.get("prec_bits", 768)),
            dps=int(d.get("dps", 220)),
            allow_currents=bool(d.get("allow_currents", False)),
            isolated_points=tuple(float(x) for x in d.get("isolated_points", ())),
        )


class ModularVerifier(Verifier):
    domain = "modular"

    def __init__(self, *, work_dir: Optional[str] = None) -> None:
        self.work_dir = Path(work_dir) if work_dir else (_REPO / "results" / "modular_work")

    def known_limits(self) -> list[str]:
        return ["modular_invariance", "unitarity", "virasoro_vacuum_c>1",
                "cL_equals_cR", "spinless_relaxation"]

    # -- public entry points ------------------------------------------------ #
    def verify(self, c: Candidate) -> VerdictGrounded:
        try:
            if isinstance(c.meta.get("modular_spec"), dict):
                spec = ModularSpec.from_dict(c.meta["modular_spec"])
            else:
                spec = ModularSpec.from_dict(json.loads(c.artifact))
        except Exception as e:  # noqa: BLE001
            return self._error(f"could not parse modular spec: {e!r}")
        return self.verify_spec(spec)

    def verify_spec(self, spec: ModularSpec) -> VerdictGrounded:
        _ensure_env()
        t0 = time.time()
        if spec.c_total <= 2:
            return self._error("c_total must be > 2 (nondegenerate Virasoro characters)")
        if spec.n_derivs < 1:
            return self._error("n_derivs must be >= 1")
        try:
            pmp = self._build_pmp(spec)
            run_dir = self.work_dir / self._dirname(spec)
            if run_dir.exists():
                shutil.rmtree(run_dir)
            with _in_dir(run_dir):
                terminate, functional = self._solve(pmp, spec)
        except Exception as e:  # noqa: BLE001
            return self._error(f"modular pipeline failed: {e!r}", cost=time.time() - t0)

        cost = time.time() - t0
        # Same 3-way soundness discipline as the other bootstrap verifiers.
        if terminate == "found primal feasible solution":
            feasible = True
        elif terminate == "found dual feasible solution":
            feasible = False
        else:
            return self._error(
                f"SDPB inconclusive: terminateReason={terminate!r} — no "
                f"feasibility certificate; raise maxIterations and re-run",
                cost=cost,
            )

        scope = ("all unitary compact c_L=c_R CFTs (current-class positivity "
                 "imposed)" if spec.allow_currents else
                 "twist-gapped theories: no conserved currents besides the "
                 "stress tensor (Virasoro-only; standard benchmark convention)")
        certificate = json.dumps({
            "kind": "sdpb_modular_gap",
            "c_total": spec.c_total, "gap": spec.gap,
            "n_derivs": spec.n_derivs, "prec_bits": spec.prec_bits,
            "allow_currents": spec.allow_currents,
            "isolated_points": list(spec.isolated_points),
            "assumptions": ("unitary compact 2D CFT, c_L=c_R, modular-invariant "
                            "partition function, no primaries below gap except "
                            f"the declared isolated states {list(spec.isolated_points)}; "
                            f"exclusion scope: {scope}; spinless relaxation"),
            "gravity_reading": ((f"excluded => every consistent AdS3 quantum gravity "
                                 f"with c_total=3l/G={spec.c_total:g} in scope "
                                 f"[{scope}] has a primary below {spec.gap:g}; "
                                 f"BTZ black-hole threshold is c_total/24="
                                 f"{spec.c_total/24:g}") if not feasible else
                                "allowed at this derivative order"),
            "feasible": feasible, "terminateReason": terminate,
            # y is only a proof object (the exclusion functional) when dual
            # feasible; for allowed verdicts it is meaningless — omit it.
            "dual_functional_y": (functional if not feasible else None),
        }, sort_keys=True)

        verdict = VerdictGrounded(
            valid=feasible,
            fitness=float(spec.gap) if feasible else -1.0,
            certificate=certificate,
            reproduces_known={"modular_invariance": True, "unitarity": True,
                              "gap_allowed": feasible},
            cost=cost,
            details={"feasible": feasible, "terminateReason": terminate,
                     "c_total": spec.c_total, "gap": spec.gap,
                     "n_derivs": spec.n_derivs},
        )
        verdict.audit()
        return verdict

    def gap_bound(self, spec: ModularSpec, lower: float, upper: float,
                  threshold: float = 0.01) -> dict:
        """Bisect to the max allowed gap at this c_total — the certified AdS3
        statement 'a primary must exist below (bound + threshold)'."""
        t0 = time.time()
        from dataclasses import replace
        lo, hi = float(lower), float(upper)
        # Verify the bisection preconditions instead of trusting them: lo must
        # be allowed and hi excluded, else the result would be meaningless.
        v_lo = self.verify_spec(replace(spec, gap=lo))
        if v_lo.error or not v_lo.valid:
            raise RuntimeError(f"gap_bound precondition failed: lower={lo} is not "
                               f"allowed (err={v_lo.error})")
        v_hi = self.verify_spec(replace(spec, gap=hi))
        if v_hi.error or v_hi.valid:
            raise RuntimeError(f"gap_bound precondition failed: upper={hi} is not "
                               f"excluded (err={v_hi.error})")
        while hi - lo > threshold:
            mid = 0.5 * (lo + hi)
            v = self.verify_spec(replace(spec, gap=mid))
            if v.error:
                raise RuntimeError(f"bisection failed at gap={mid}: {v.error}")
            if v.valid:
                lo = mid
            else:
                hi = mid
        return {"c_total": spec.c_total, "gap_bound_allowed": lo,
                "gap_bound_excluded_above": hi, "n_derivs": spec.n_derivs,
                "cost_s": round(time.time() - t0, 1)}

    # -- PMP construction ---------------------------------------------------- #
    def _dirname(self, spec: ModularSpec) -> str:
        import hashlib
        h = hashlib.sha1(json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()[:10]
        return f"pt_c{spec.c_total:g}_gap{spec.gap:.6f}_{h}"

    def _build_pmp(self, spec: ModularSpec) -> dict:
        """PMP JSON: find z (functional) with n·z = 1 (vacuum normalization) and
        Σ_k z_k Q_k(x* + t) ≥ 0 for t ∈ [0, ∞)  (prefactor e^{-2πt} > 0)."""
        mp.mp.dps = spec.dps
        A = (mp.mpf(spec.c_total) - 2) / 24          # x = Δ - A
        x_star = mp.mpf(spec.gap) - A                 # continuum start
        orders = [2 * k + 1 for k in range(spec.n_derivs)]

        # exact -> mpf coefficient lists per functional component
        q_mpf = []
        for m in orders:
            exact = _q_poly_coeffs(m)
            q_mpf.append([mp.mpf(str(sp.N(cf, spec.dps + 10))) for cf in exact])

        # normalization: alpha(vacuum), vacuum = states at Δ=0,1,2 weights (1,-2,1)
        norm = []
        for coeffs in q_mpf:
            tot = mp.mpf(0)
            for k, w in ((0, 1), (1, -2), (2, 1)):
                xv = mp.mpf(k) - A
                tot += w * mp.e ** (-2 * mp.pi * xv) * _poly_eval(coeffs, xv)
            norm.append(tot)

        # positivity polynomials: shift Q(x* + t), expand in t (exact binomials)
        shifted = [_poly_shift(coeffs, x_star) for coeffs in q_mpf]

        def s(v):  # mpf -> decimal string
            return mp.nstr(v, spec.dps, strip_zeros=False)

        def block(polys):
            return {
                "DampedRational": {"constant": "1",
                                   "base": s(mp.e ** (-2 * mp.pi)), "poles": []},
                "polynomials": [[[  # 1x1 matrix; vector over functional comps
                    [s(cf) for cf in comp] for comp in polys
                ]]],
            }

        blocks = [block(shifted)]
        for pt in spec.isolated_points:
            # single-point positivity α[Δ_i] ≥ 0: a degree-0 block with constant
            # "polynomials" equal to e^{-2π x_i} Q_k(x_i) (prefactor trivial).
            xi = mp.mpf(pt) - A
            wi = mp.e ** (-2 * mp.pi * xi)
            const_vec = [[wi * _poly_eval(coeffs, xi)] for coeffs in q_mpf]
            blocks.append({
                "DampedRational": {"constant": "1", "base": s(mp.e ** (-2 * mp.pi)),
                                   "poles": []},
                "polynomials": [[[ [s(cv[0])] for cv in const_vec ]]],
            })
        if spec.allow_currents:
            # Conserved-current class: degenerate character χ̂(Δ)−χ̂(Δ+1) gives
            # e^{-2πx}[Q(x) − e^{-2π} Q(x+1)] — one more polynomial positivity
            # class (continuous-spin relaxation of integer J ≥ gap: rigorous).
            e2pi = mp.e ** (-2 * mp.pi)
            shifted1 = [_poly_shift(coeffs, x_star + 1) for coeffs in q_mpf]
            current = [[a - e2pi * b for a, b in zip(c0, c1)]
                       for c0, c1 in zip(shifted, shifted1)]
            blocks.append(block(current))

        return {
            "objective": ["0"] * spec.n_derivs,
            "normalization": [s(n) for n in norm],
            "PositiveMatrixWithPrefactorArray": blocks,
        }

    # -- solve ---------------------------------------------------------------- #
    def _solve(self, pmp: dict, spec: ModularSpec):
        # This verifier needs pmp2sdp, which the amd64 fallback image
        # (wlandry/sdpb:2.5.1) lacks. bootstrapcollaboration/sdpb:3.0.0 is
        # multi-arch (amd64+arm64), so it is the right default on EVERY host.
        os.environ.setdefault("QGSE_SDPB_IMAGE", "bootstrapcollaboration/sdpb:3.0.0")
        if "wlandry" in os.environ.get("QGSE_SDPB_IMAGE", ""):
            raise RuntimeError(
                "modular verifier requires pmp2sdp; QGSE_SDPB_IMAGE points at a "
                "wlandry/sdpb image which does not ship it — use "
                "bootstrapcollaboration/sdpb:3.0.0")
        Path("pmp.json").write_text(json.dumps(pmp))
        nproc = os.environ.get("QGSE_SDPB_NPROC", "1")
        prec = str(spec.prec_bits)
        with open("solver.log", "w") as log:
            subprocess.check_call(["mpirun", "-n", nproc, "pmp2sdp", "-f", "json",
                                   "-i", "pmp.json", "-o", "qgse_modular", "-p", prec],
                                  stdout=log, stderr=subprocess.STDOUT)
            subprocess.check_call(["mpirun", "-n", nproc, "sdpb", "-s", "qgse_modular",
                                   f"--precision={prec}",
                                   "--findPrimalFeasible", "--findDualFeasible"],
                                  stdout=log, stderr=subprocess.STDOUT)
        out = Path("qgse_modular_out/out.txt")
        terminate = "unknown"
        if out.exists():
            for line in out.read_text().splitlines():
                if "terminateReason" in line:
                    terminate = line.split("=", 1)[1].strip().strip('";')
        functional = None
        ytxt = Path("qgse_modular_out/y.txt")
        if ytxt.exists():
            # format: "rows cols" header then rows values; with a normalization
            # vector y has (n_derivs - 1) entries. Parse the header — the old
            # tail-slice swallowed a header token into the certificate (review
            # finding; verdicts unaffected, proof object now correct).
            vals = ytxt.read_text().split()
            try:
                rows = int(vals[0])
                functional = vals[2:2 + rows]
            except Exception:  # noqa: BLE001
                functional = vals
        # keep out dir (certificate artifact), drop converted blocks/checkpoints
        from qgse.verifiers.bootstrap_mixed import _cleanup_solver_files
        _cleanup_solver_files("qgse_modular")
        Path("pmp.json").unlink(missing_ok=True)
        return terminate, functional

    def _error(self, msg: str, cost: float = 0.0) -> VerdictGrounded:
        return VerdictGrounded(valid=False, fitness=-1e9, certificate=None,
                               reproduces_known={}, cost=cost, error=msg)


# --------------------------------------------------------------------------- #
# mpmath polynomial helpers (coefficients lowest-degree first)
# --------------------------------------------------------------------------- #
def _poly_eval(coeffs, x):
    tot = mp.mpf(0)
    for c in reversed(coeffs):
        tot = tot * x + c
    return tot


def _poly_shift(coeffs, a):
    """Coefficients of P(a + t) in t, exact binomial expansion in mpmath."""
    n = len(coeffs)
    out = [mp.mpf(0)] * n
    for j, cj in enumerate(coeffs):
        for k in range(j + 1):
            out[k] += cj * mp.binomial(j, k) * a ** (j - k)
    return out
