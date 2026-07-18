"""Explicit General-Relativity verifier (SymPy).

A candidate is a metric ansatz. This verifier computes the curvature tensors
from scratch (Christoffel -> Riemann -> Ricci -> Einstein) and checks, with an
auditable certificate:

    * Einstein's equations. In VACUUM mode it requires G_ab + Λ g_ab = 0
      (symbolic proof, or a coordinate-INVARIANT numeric residual). In MATTER
      mode any metric defines *some* stress-energy T_ab, so the equations hold by
      construction and the meaningful discriminator is the **energy conditions**
      (null/weak/strong/dominant) — which therefore GATE validity, not decorate
      it.
    * Admissibility, checked CHEAPLY first (parse, non-degeneracy, Lorentzian
      signature at sample points) so inadmissible candidates are rejected BEFORE
      the expensive symbolic curvature pass.
    * Correct limits: flat spacetime when the source parameter -> 0; asymptotics
      matched via the coordinate-invariant Kretschmann scalar K = R_abcd R^abcd.
    * Structure: a Killing/event horizon (roots of g^rr, so rotating metrics
      whose horizon is not at g_tt = 0 are handled).

Nothing here consults an LLM. Fitness is built only from residuals / feasibility
margins, all coordinate-invariant. The certificate is a JSON blob an independent
checker can re-verify (§9). Core entry: :meth:`GRVerifier.verify_spec`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sympy as sp

from qgse.interfaces import Candidate, Verifier, VerdictGrounded


# --------------------------------------------------------------------------- #
# Metric specification
# --------------------------------------------------------------------------- #
@dataclass
class MetricSpec:
    """A metric ansatz in a chosen coordinate chart.

    coords: coordinate names, e.g. ``["t", "r", "theta", "phi"]``.
    g:      full n x n metric, entries as SymPy-parseable strings.
    params: parameter symbol -> assumption ("positive" | "real" | "nonzero").
    matter: "vacuum" (require G_ab + Λ g_ab = 0) or "matter" (derive T_ab and
            require the energy conditions).
    Lambda: cosmological constant string. G: Newton's constant string.
    signature: "mostly_plus" (-+++) or "mostly_minus" (+---).
    asymptotics: "flat" | "ads" | "ds" | "none".
    coord_roles: coord-name -> {"time","radial","angle","azimuth","generic"}.
    source_param: parameter whose ->0 limit should recover flat spacetime.
    energy_conditions_required: which ECs gate validity in matter mode
                                (default null+weak, i.e. NEC and WEC).
    """

    coords: list[str]
    g: list[list[str]]
    params: dict[str, str] = field(default_factory=dict)
    matter: str = "vacuum"
    Lambda: str = "0"
    G: str = "1"
    signature: str = "mostly_plus"
    asymptotics: str = "flat"
    coord_roles: dict[str, str] = field(default_factory=dict)
    source_param: Optional[str] = None
    energy_conditions_required: list[str] = field(
        default_factory=lambda: ["null", "weak"]
    )

    @staticmethod
    def from_dict(d: dict) -> "MetricSpec":
        return MetricSpec(
            coords=list(d["coords"]),
            g=[[str(x) for x in row] for row in d["g"]],
            params=dict(d.get("params", {})),
            matter=str(d.get("matter", "vacuum")),
            Lambda=str(d.get("Lambda", "0")),
            G=str(d.get("G", "1")),
            signature=str(d.get("signature", "mostly_plus")),
            asymptotics=str(d.get("asymptotics", "flat")),
            coord_roles=dict(d.get("coord_roles", {})),
            source_param=d.get("source_param"),
            energy_conditions_required=list(
                d.get("energy_conditions_required", ["null", "weak"])
            ),
        )


class SpecError(ValueError):
    """The candidate did not describe a well-formed metric."""


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
class GRVerifier(Verifier):
    domain = "gr"

    def __init__(
        self,
        *,
        numeric_efe_tol: float = 1e-5,  # threshold on the bounded invariant rnorm
        n_samples: int = 24,
        seed: int = 12345,
        max_dim: int = 6,
        disable_symbolic: bool = False,  # skip the symbolic proof -> force numeric path
    ) -> None:
        self.numeric_efe_tol = numeric_efe_tol
        self.n_samples = n_samples
        self.seed = seed
        self.max_dim = max_dim
        self.disable_symbolic = disable_symbolic

    def known_limits(self) -> list[str]:
        """Admissibility constraints. The first three are checked CHEAPLY (no
        curvature) and short-circuit rejection. For classical GR the ℏ->0 limit
        is vacuous; the computable checks are these."""
        return [
            "nondegenerate",           # det g != 0            (cheap)
            "symmetric",               # g_ab = g_ba           (cheap)
            "lorentzian_signature",    # one timelike axis     (cheap)
            "satisfies_efe",           # vacuum: G_ab+Λg_ab=0; matter: energy conds
            "recovers_flat_no_source", # source_param -> 0 gives flat spacetime
            "asymptotics_match",       # large-r curvature matches target
        ]

    # -- public entry points ------------------------------------------------ #
    def verify(self, c: Candidate) -> VerdictGrounded:
        try:
            if isinstance(c.meta.get("metric_spec"), dict):
                spec = MetricSpec.from_dict(c.meta["metric_spec"])
            else:
                spec = MetricSpec.from_dict(json.loads(c.artifact))
        except SpecError as e:
            return self._error(str(e))
        except Exception as e:  # noqa: BLE001
            return self._error(f"could not parse metric spec: {e!r}")
        return self.verify_spec(spec)

    def verify_spec(self, spec: MetricSpec) -> VerdictGrounded:
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        try:
            tens = _Tensors(spec, max_dim=self.max_dim)  # parse only, no curvature
        except SpecError as e:
            return self._error(str(e), cost=time.time() - t0)
        except Exception as e:  # noqa: BLE001
            return self._error(f"tensor construction failed: {e!r}", cost=time.time() - t0)

        # ---- CHEAP admissibility gate (no curvature) --------------------- #
        try:
            samples = tens.sample_points(self.n_samples, rng)
            cheap = tens.cheap_gate(samples)
        except Exception as e:  # noqa: BLE001
            return self._error(f"cheap admissibility failed: {e!r}", cost=time.time() - t0)

        if not cheap["any_good"]:
            return self._error(
                "metric could not be evaluated at any sample point "
                "(coordinate singularities or undefined functions?)",
                cost=time.time() - t0,
            )

        symmetric = tens.is_symmetric()
        nondegenerate = cheap["nondegenerate"]
        want = (1, tens.n - 1) if spec.signature == "mostly_plus" else (tens.n - 1, 1)
        lorentzian = cheap["signature_mode"] == want
        admissible = symmetric and nondegenerate and lorentzian

        if not admissible:
            return self._inadmissible_verdict(
                spec, symmetric, nondegenerate, lorentzian,
                cheap["signature_mode"], time.time() - t0,
            )

        # ---- admissible: now pay for curvature --------------------------- #
        try:
            tens.ensure_curvature()
            report = self._analyze(spec, tens, cheap, samples, rng)
        except Exception as e:  # noqa: BLE001
            return self._error(f"curvature analysis failed: {e!r}", cost=time.time() - t0)

        return self._assemble(spec, report, symmetric, nondegenerate, lorentzian,
                              cheap["signature_mode"], time.time() - t0)

    # -- verdict assembly --------------------------------------------------- #
    def _assemble(self, spec, report, symmetric, nondegenerate, lorentzian,
                  sig_mode, cost) -> VerdictGrounded:
        is_vacuum = spec.matter == "vacuum"

        if is_vacuum:
            efe_ok = report["symbolic_zero"] or report["rnorm"] < self.numeric_efe_tol
        else:
            efe_ok = report["energy_conditions_ok"]  # matter: ECs gate validity

        valid = symmetric and nondegenerate and lorentzian and efe_ok

        reproduces = {
            "nondegenerate": nondegenerate,
            "symmetric": symmetric,
            "lorentzian_signature": lorentzian,
            "satisfies_efe": efe_ok if is_vacuum else True,  # matter: EFE hold by construction
            "recovers_flat_no_source": report["recovers_flat"],
            "asymptotics_match": report["asymptotics_ok"],
        }
        if not is_vacuum:
            reproduces["energy_conditions_ok"] = report["energy_conditions_ok"]
            reproduces.update({f"ec_{k}": v for k, v in report["energy_conditions"].items()
                               if k in ("null", "weak", "strong", "dominant")})

        fitness = self._fitness(spec, report, symmetric, nondegenerate,
                                lorentzian, valid)

        details = {
            "efe_residual_invariant": report["rnorm"],
            "efe_symbolic_zero": report["symbolic_zero"],
            "kretschmann_trend": report["kretschmann_trend"],
            "asymptotics_detected": report["asymptotics_detected"],
            "horizon": report["horizon"],
            "signature": sig_mode,
            "n_good_samples": report["n_good"],
        }
        if not is_vacuum:
            details["energy_conditions"] = report["energy_conditions"]

        certificate = json.dumps(report["certificate"], sort_keys=True) if valid else None
        if not valid:
            details["certificate_preview"] = report["certificate"].get("summary", "")

        verdict = VerdictGrounded(
            valid=valid,
            fitness=fitness,
            certificate=certificate,
            reproduces_known=reproduces,
            cost=cost,
            details=details,
            private={"full_certificate": report["certificate"]},
        )
        verdict.audit()
        return verdict

    def _inadmissible_verdict(self, spec, symmetric, nondegenerate, lorentzian,
                              sig_mode, cost) -> VerdictGrounded:
        """Reject a candidate that failed the CHEAP gate — without ever building
        curvature. Fitness rewards only the admissibility partials achieved."""
        score = 0.0
        if nondegenerate:
            score += 5.0
        if symmetric:
            score += 5.0
        if lorentzian:
            score += 10.0
        reasons = []
        if not symmetric:
            reasons.append("asymmetric metric")
        if not nondegenerate:
            reasons.append("degenerate metric (det g = 0)")
        if not lorentzian:
            reasons.append(f"non-Lorentzian signature {sig_mode}")
        return VerdictGrounded(
            valid=False,
            fitness=score,
            certificate=None,
            reproduces_known={
                "nondegenerate": nondegenerate,
                "symmetric": symmetric,
                "lorentzian_signature": lorentzian,
                "satisfies_efe": False,
            },
            cost=cost,
            error="inadmissible: " + "; ".join(reasons),
            details={"signature": sig_mode, "cheap_rejected": True},
        )

    def _error(self, msg: str, cost: float = 0.0) -> VerdictGrounded:
        return VerdictGrounded(
            valid=False, fitness=-1e9, certificate=None, reproduces_known={},
            cost=cost, error=msg,
        )

    # -- analysis (post-curvature) ------------------------------------------ #
    def _analyze(self, spec, tens, cheap, samples, rng) -> dict:
        is_vacuum = spec.matter == "vacuum"
        symbolic_zero = (
            tens.symbolic_residual_zero()
            if (is_vacuum and not self.disable_symbolic)
            else False
        )

        # coordinate-invariant residual norm + Kretschmann at each good sample
        rnorms, ks = [], []
        for pt in cheap["good_pts"]:
            inv = tens.invariants_at(pt)
            if inv is None:
                continue
            rnorms.append(inv["rnorm"])
            ks.append(inv["K"])
        if not rnorms:
            raise SpecError("curvature invariants could not be evaluated")
        rnorm = float(np.median(rnorms))

        # energy conditions only in matter mode (wasted on ~0 vacuum T)
        if is_vacuum:
            energy_conditions = {"note": "vacuum: energy conditions trivial"}
            ec_ok = True
        else:
            energy_conditions = tens.energy_conditions(cheap["good_pts"][:12])
            required = spec.energy_conditions_required or ["null", "weak"]
            ec_ok = all(bool(energy_conditions.get(c)) for c in required)

        recovers_flat, flat_detail = tens.recovers_flat_no_source(rng)
        asymptotics_detected, k_trend, asym_ok = tens.asymptotics(spec.asymptotics, rng)
        horizon = tens.detect_horizon()

        # Einstein tensor for the certificate: symbolic (exact strings) when the
        # proof fast-path built it, else a numeric snapshot at a sample point.
        if getattr(tens, "_symbolic_ok", False):
            einstein_cert = tens.einstein_str()
            ricci_cert = str(tens.Rscalar)
        else:
            cur0 = tens._curvature_at(cheap["good_pts"][0])
            einstein_cert = (np.round(cur0["Ein"], 8).tolist()
                             if cur0 is not None else "unavailable")
            ricci_cert = (round(cur0["Rscalar"], 8) if cur0 is not None else None)

        certificate = {
            "summary": (
                f"n={tens.n} coords={spec.coords} matter={spec.matter} "
                f"Lambda={spec.Lambda} rnorm={rnorm:.3e} "
                f"symbolic_zero={symbolic_zero}"
            ),
            "coords": spec.coords,
            "metric": [[str(x) for x in row] for row in tens.g_syms],
            "params": dict(spec.params),          # store REAL params (audit uses these)
            "Lambda": spec.Lambda,
            "G": spec.G,
            "matter": spec.matter,
            "signature": spec.signature,
            "einstein_tensor_lower": einstein_cert,
            "ricci_scalar": ricci_cert,
            "residual_norm_invariant": rnorm,
            "numeric_efe_tol": self.numeric_efe_tol,
            "symbolic_residual_zero": symbolic_zero,
            "n_good_samples": len(rnorms),
            "recovers_flat_no_source": {"ok": recovers_flat, **flat_detail},
            "asymptotics": {"target": spec.asymptotics, "detected": asymptotics_detected,
                            "kretschmann_trend": k_trend, "match": asym_ok},
            "horizon": horizon,
            "energy_conditions": energy_conditions,
        }

        return {
            "rnorm": rnorm,
            "symbolic_zero": symbolic_zero,
            "energy_conditions": energy_conditions,
            "energy_conditions_ok": ec_ok,
            "recovers_flat": recovers_flat,
            "asymptotics_ok": asym_ok,
            "asymptotics_detected": asymptotics_detected,
            "kretschmann_trend": k_trend,
            "horizon": horizon,
            "n_good": len(rnorms),
            "certificate": certificate,
        }

    def _fitness(self, spec, report, symmetric, nondegenerate, lorentzian, valid) -> float:
        """Grounded scalar (maximized), built only from invariants. Admissible
        base + (vacuum) EFE closeness or (matter) energy conditions + structure."""
        score = 0.0
        if nondegenerate:
            score += 5.0
        if symmetric:
            score += 5.0
        if lorentzian:
            score += 10.0

        if spec.matter == "vacuum":
            if report["symbolic_zero"]:
                score += 120.0 + 50.0  # max EFE score + proof bonus
            else:
                efe_score = min(12.0, -math.log10(report["rnorm"] + 1e-12))
                score += 10.0 * efe_score
        else:
            ec = report["energy_conditions"]
            for cond in ("null", "weak", "strong", "dominant"):
                if ec.get(cond):
                    score += 15.0

        if report["recovers_flat"]:
            score += 10.0
        if report["asymptotics_ok"]:
            score += 10.0
        if report["horizon"].get("has_horizon"):
            score += 8.0
        if valid:
            score += 100.0
        return float(score)


# --------------------------------------------------------------------------- #
# Tensor engine
# --------------------------------------------------------------------------- #
class _Tensors:
    """Symbolic curvature machinery + numeric (invariant) evaluation. Curvature
    is built lazily via :meth:`ensure_curvature` so the cheap admissibility gate
    can run first."""

    def __init__(self, spec: MetricSpec, max_dim: int = 6) -> None:
        self.spec = spec
        n = len(spec.coords)
        self.n = n
        if n < 2 or n > max_dim:
            raise SpecError(f"unsupported dimension n={n} (allowed 2..{max_dim})")
        if len(spec.g) != n or any(len(row) != n for row in spec.g):
            raise SpecError(f"metric must be {n}x{n} to match {n} coordinates")

        self.coord_syms = [self._coord_symbol(name) for name in spec.coords]
        self.param_syms: dict[str, sp.Symbol] = {}
        for name, tag in spec.params.items():
            self.param_syms[name] = self._param_symbol(name, tag)

        self.local = {}
        for name, s in zip(spec.coords, self.coord_syms):
            self.local[name] = s
        self.local.update(self.param_syms)
        for fn in ("sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh"):
            self.local.setdefault(fn, getattr(sp, fn))
        self.local.setdefault("pi", sp.pi)

        try:
            g_syms = [[self._parse(spec.g[a][b]) for b in range(n)] for a in range(n)]
        except Exception as e:  # noqa: BLE001
            raise SpecError(f"could not parse metric entry: {e!r}")
        self.g_syms = g_syms
        self.g = sp.Matrix(g_syms)
        self.Lambda = self._parse(spec.Lambda)
        self.Gnewton = self._parse(spec.G)

        allowed = set(self.coord_syms) | set(self.param_syms.values())
        free = self.g.free_symbols | self.Lambda.free_symbols | self.Gnewton.free_symbols
        extra = {s for s in free if s not in allowed}
        if extra:
            raise SpecError(
                "metric references undeclared symbols "
                f"{sorted(str(s) for s in extra)}; declare them in params "
                "or write explicit functions of the coordinates"
            )

        self._curvature_built = False
        # cheap metric-only lambdify
        self._fn_g = sp.lambdify(self._syms(), self.g, modules="numpy")

    # -- symbols ------------------------------------------------------------ #
    def _syms(self):
        return self.coord_syms + list(self.param_syms.values())

    def _coord_symbol(self, name: str) -> sp.Symbol:
        role = self.spec.coord_roles.get(name) or _infer_role(name)
        if role == "radial":
            return sp.Symbol(name, positive=True)
        return sp.Symbol(name, real=True)

    def _param_symbol(self, name: str, tag: str) -> sp.Symbol:
        tag = (tag or "real").lower()
        if tag == "positive":
            return sp.Symbol(name, positive=True)
        if tag == "nonzero":
            return sp.Symbol(name, nonzero=True, real=True)
        return sp.Symbol(name, real=True)

    def _parse(self, expr: str) -> sp.Expr:
        return sp.sympify(expr, locals=self.local, rational=True)

    # -- curvature (lazy, NUMERIC-FIRST) ------------------------------------ #
    def ensure_curvature(self) -> None:
        """Build curvature evaluators. Numeric-first: we lambdify only the
        metric's first and second derivatives (cheap — no Christoffel-product
        expression swell), then do ALL tensor algebra numerically at sample
        points via :meth:`_curvature_at`. This is universal (handles Kerr and
        other heavy off-diagonal metrics). The SymPy proof is an OPTIONAL
        fast-path, built only for simple (diagonal) metrics where it is cheap."""
        if self._curvature_built:
            return
        n, g, x = self.n, self.g, self.coord_syms
        syms = self._syms()
        dg = [[[sp.diff(g[a, b], x[c]) for b in range(n)] for a in range(n)]
              for c in range(n)]                                     # dg[c][a][b]
        d2g = [[[[sp.diff(g[a, b], x[c], x[d]) for b in range(n)] for a in range(n)]
                for d in range(n)] for c in range(n)]                # d2g[c][d][a][b]
        self._fn_dg = sp.lambdify(syms, dg, modules="numpy")
        self._fn_d2g = sp.lambdify(syms, d2g, modules="numpy")
        self._fn_Lambda = sp.lambdify(syms, self.Lambda, modules="numpy")
        self._fn_G = sp.lambdify(syms, self.Gnewton, modules="numpy")

        # Optional symbolic proof fast-path (diagonal metrics only).
        self._symbolic_ok = False
        if self._is_diagonal():
            try:
                self._build_symbolic()
                self._symbolic_ok = True
            except Exception:  # noqa: BLE001
                self._symbolic_ok = False
        self._curvature_built = True

    def _is_diagonal(self) -> bool:
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.g_syms[i][j] != 0:
                    try:
                        if sp.simplify(self.g_syms[i][j]) != 0:
                            return False
                    except Exception:  # noqa: BLE001
                        return False
        return True

    def _build_symbolic(self) -> None:
        """Symbolic Christoffel->Riemann->Einstein->residual for the proof
        fast-path. Only invoked for simple metrics (kept small)."""
        n, g, x = self.n, self.g, self.coord_syms
        ginv = g.inv()
        Gamma = [[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(b, n):
                    s = sp.Integer(0)
                    for d in range(n):
                        s += ginv[a, d] * (sp.diff(g[d, c], x[b])
                                           + sp.diff(g[d, b], x[c])
                                           - sp.diff(g[b, c], x[d]))
                    val = sp.Rational(1, 2) * s
                    Gamma[a][b][c] = val
                    Gamma[a][c][b] = val
        Rie = [[[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        term = sp.diff(Gamma[a][b][d], x[c]) - sp.diff(Gamma[a][b][c], x[d])
                        for e in range(n):
                            term += Gamma[a][c][e] * Gamma[e][b][d]
                            term -= Gamma[a][d][e] * Gamma[e][b][c]
                        Rie[a][b][c][d] = term
        Ric = sp.zeros(n, n)
        for b in range(n):
            for d in range(n):
                s = sp.Integer(0)
                for a in range(n):
                    s += Rie[a][b][a][d]
                Ric[b, d] = s
        R = sp.Integer(0)
        for b in range(n):
            for d in range(n):
                R += ginv[b, d] * Ric[b, d]
        self.Rscalar = R
        self.Ein = sp.Matrix(n, n, lambda i, j: Ric[i, j] - sp.Rational(1, 2) * R * g[i, j])
        self.residual = sp.Matrix(n, n, lambda i, j: self.Ein[i, j] + self.Lambda * g[i, j])

    def _curvature_at(self, pt) -> Optional[dict]:
        """Numeric curvature at one point from {g, ∂g, ∂²g}. No symbolic
        Christoffel/Riemann is ever formed, so this is fast and universal."""
        vec = self._vec(pt)
        try:
            g = np.asarray(self._fn_g(*vec), dtype=float)
            dg = np.asarray(self._fn_dg(*vec), dtype=float)      # [c,a,b] = ∂_c g_ab
            d2g = np.asarray(self._fn_d2g(*vec), dtype=float)    # [c,d,a,b]
            Lam = float(self._fn_Lambda(*vec))
            Gn = float(self._fn_G(*vec))
        except Exception:  # noqa: BLE001
            return None
        n = self.n
        if (g.shape != (n, n) or dg.shape != (n, n, n)
                or d2g.shape != (n, n, n, n)):
            return None
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(dg))
                and np.all(np.isfinite(d2g))):
            return None
        try:
            ginv = np.linalg.inv(g)
        except np.linalg.LinAlgError:
            return None

        Gamma = np.zeros((n, n, n))  # Γ^a_{bc}
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    s = 0.0
                    for d in range(n):
                        s += ginv[a, d] * (dg[b, d, c] + dg[c, d, b] - dg[d, b, c])
                    Gamma[a, b, c] = 0.5 * s
        dginv = np.zeros((n, n, n))  # ∂_e g^{ad} = -g^{ap} g^{dq} ∂_e g_{pq}
        for e in range(n):
            for a in range(n):
                for d in range(n):
                    s = 0.0
                    for p in range(n):
                        for q in range(n):
                            s += ginv[a, p] * ginv[d, q] * dg[e, p, q]
                    dginv[e, a, d] = -s
        dGamma = np.zeros((n, n, n, n))  # ∂_e Γ^a_{bc}
        for e in range(n):
            for a in range(n):
                for b in range(n):
                    for c in range(n):
                        s = 0.0
                        for d in range(n):
                            paren = dg[b, d, c] + dg[c, d, b] - dg[d, b, c]
                            dparen = d2g[e, b, d, c] + d2g[e, c, d, b] - d2g[e, d, b, c]
                            s += dginv[e, a, d] * paren + ginv[a, d] * dparen
                        dGamma[e, a, b, c] = 0.5 * s
        Rup = np.zeros((n, n, n, n))  # R^a_{bcd}
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        s = dGamma[c, a, b, d] - dGamma[d, a, b, c]
                        for e in range(n):
                            s += Gamma[a, c, e] * Gamma[e, b, d]
                            s -= Gamma[a, d, e] * Gamma[e, b, c]
                        Rup[a, b, c, d] = s
        Ric = np.einsum("abad->bd", Rup)
        Rs = float(np.einsum("bd,bd->", ginv, Ric))
        Ein = Ric - 0.5 * Rs * g
        residual = Ein + Lam * g
        T = residual / (8.0 * np.pi * Gn)
        Rlow = np.einsum("ae,ebcd->abcd", g, Rup)
        Rupper = np.einsum("ai,bj,ck,dl,ijkl->abcd", ginv, ginv, ginv, ginv, Rlow)
        K = float(np.einsum("abcd,abcd->", Rlow, Rupper))
        return {"g": g, "ginv": ginv, "Ein": Ein, "residual": residual,
                "T": T, "K": K, "Ric": Ric, "Rscalar": Rs}

    # -- string exports ----------------------------------------------------- #
    def einstein_str(self) -> list[list[str]]:
        return [[str(self.Ein[i, j]) for j in range(self.n)] for i in range(self.n)]

    def is_symmetric(self) -> bool:
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if sp.simplify(self.g[i, j] - self.g[j, i]) != 0:
                    return False
        return True

    def symbolic_residual_zero(self) -> bool:
        if not getattr(self, "_symbolic_ok", False):
            return False  # proof fast-path unavailable -> numeric path decides
        try:
            for i in range(self.n):
                for j in range(i, self.n):
                    if sp.simplify(self.residual[i, j]) != 0:
                        return False
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- numeric sampling --------------------------------------------------- #
    def _param_values(self, rng) -> dict:
        vals = {}
        for name, s in self.param_syms.items():
            vals[name] = float(rng.uniform(0.5, 2.0)) if s.is_positive else \
                (float(rng.uniform(-2.0, 2.0)) or 0.7)
        return vals

    def _coord_values(self, rng, params) -> dict:
        vals = {}
        pscale = max([abs(v) for v in params.values()] + [1.0])
        for name in self.spec.coords:
            role = self.spec.coord_roles.get(name) or _infer_role(name)
            if role == "time":
                vals[name] = float(rng.uniform(-1.0, 1.0))
            elif role == "radial":
                vals[name] = float(rng.uniform(4.0 * pscale, 12.0 * pscale))
            elif role == "angle":
                vals[name] = float(rng.uniform(0.4, math.pi - 0.4))
            elif role == "azimuth":
                vals[name] = float(rng.uniform(0.1, 2 * math.pi - 0.1))
            else:
                vals[name] = float(rng.uniform(0.5, 2.5))
        return vals

    def sample_points(self, n_samples, rng) -> list:
        pts = []
        for _ in range(n_samples):
            params = self._param_values(rng)
            pts.append({**params, **self._coord_values(rng, params)})
        return pts

    def _vec(self, pt) -> list:
        order = self.spec.coords + list(self.param_syms.keys())
        return [float(pt[name]) for name in order]

    def _eval(self, fn, pt, shape):
        try:
            arr = np.asarray(fn(*self._vec(pt)), dtype=float)
        except Exception:  # noqa: BLE001
            return None
        if arr.shape != shape or not np.all(np.isfinite(arr)):
            return None
        return arr

    # -- CHEAP gate: metric-only signature / nondegeneracy ------------------ #
    def cheap_gate(self, samples) -> dict:
        good_pts, sigs, detgs = [], [], []
        for pt in samples:
            g = self._eval(self._fn_g, pt, (self.n, self.n))
            if g is None:
                continue
            detg = float(np.linalg.det(g))
            if not np.isfinite(detg):
                continue
            try:
                eig = np.linalg.eigvalsh(0.5 * (g + g.T))
            except np.linalg.LinAlgError:
                continue
            good_pts.append(pt)
            detgs.append(detg)
            sigs.append((int(np.sum(eig < 0)), int(np.sum(eig > 0))))
        return {
            "any_good": len(good_pts) > 0,
            "good_pts": good_pts,
            "nondegenerate": bool(detgs) and all(abs(d) > 1e-12 for d in detgs),
            "signature_mode": _mode(sigs),
            "signatures": sigs,
        }

    # -- coordinate-INVARIANT residual + Kretschmann ------------------------ #
    def invariants_at(self, pt) -> Optional[dict]:
        cur = self._curvature_at(pt)
        if cur is None:
            return None
        E, ginv, K = cur["residual"], cur["ginv"], cur["K"]
        # invariant residual norm: sqrt(E_ab E^ab), E^ab = g^ac g^bd E_cd
        E_up = ginv @ E @ ginv
        res_inv = math.sqrt(abs(float(np.sum(E * E_up))))
        scale = math.sqrt(abs(K))
        # Normalize by the curvature scale (coordinate-invariant). The absolute
        # FLOOR handles FLAT space, where both res_inv and sqrt|K| sit at the
        # numerical noise floor (~1e-16) and their bare ratio would be a
        # meaningless O(1); with the floor, flat vacuum reads ~0 as it should,
        # while any real curved solution/non-solution is unaffected (its scale
        # >> floor). Scale-invariant for curved spacetimes; flat-safe.
        rnorm = res_inv / max(scale, res_inv, 1e-8)  # dimensionless, in [0,1]
        # NOTE (semantics): rnorm measures equation-violation RELATIVE to the
        # curvature scale sqrt|K|. A highly-curved near-miss (large Weyl, small
        # Ricci residual) scores a forgivingly small rnorm — it genuinely is
        # close to solving at its own scale. K is invariant, so there is no
        # coordinate exploit; this just mildly favors highly-curved near-misses.
        return {"rnorm": rnorm, "res_inv": res_inv, "K": K}

    def kretschmann_at(self, pt) -> Optional[float]:
        cur = self._curvature_at(pt)
        return None if cur is None else cur["K"]

    # -- flat limit as source -> 0 ------------------------------------------ #
    def recovers_flat_no_source(self, rng) -> tuple:
        src = self.spec.source_param
        if src is None:
            for name, s in self.param_syms.items():
                if s.is_positive:
                    src = name
                    break
        if src is None or src not in self.param_syms:
            return True, {"note": "no source parameter; check skipped"}

        params = self._param_values(rng)
        coords = self._coord_values(rng, params)
        pt = {**params, **coords}
        ks = []
        for eps in (1.0, 1e-1, 1e-2, 1e-3):
            pt2 = dict(pt)
            pt2[src] = eps
            k = self.kretschmann_at(pt2)
            ks.append(k if k is not None else float("nan"))
        finite = [k for k in ks if math.isfinite(k)]
        ok = (len(finite) >= 2 and abs(finite[-1]) < 1e-6
              and abs(finite[-1]) <= abs(finite[0]) + 1e-9)
        return bool(ok), {"source": src, "kretschmann_vs_source": ks}

    # -- asymptotics via Kretschmann trend ---------------------------------- #
    def asymptotics(self, target, rng) -> tuple:
        radial = _find_role(self.spec.coords, self.spec.coord_roles, "radial")
        if radial is None or target == "none":
            return "n/a", [], True
        rname = self.spec.coords[radial]
        params = self._param_values(rng)
        base = self._coord_values(rng, params)
        radii = [10.0, 100.0, 1000.0, 1e4]
        ks = []
        for r in radii:
            k = self.kretschmann_at({**params, **base, rname: r})
            ks.append(k if k is not None else float("nan"))
        finite = [(r, k) for r, k in zip(radii, ks) if math.isfinite(k)]
        if len(finite) < 2:
            return "indeterminate", ks, target == "none"
        last, first = finite[-1][1], finite[0][1]
        decreasing = abs(last) <= abs(first) + 1e-12
        if abs(last) < 1e-8 and decreasing:
            detected = "flat"
        elif decreasing and abs(last) > 1e-8:
            detected = "constant_curvature"
        else:
            detected = "non_asymptotic"
        if target == "flat":
            ok = detected == "flat"
        elif target in ("ads", "ds"):
            ok = detected == "constant_curvature"
        else:
            ok = True
        return detected, ks, bool(ok)

    def _symbolic_ginv(self):
        """Lazy symbolic metric inverse for horizon detection. The metric
        inverse is cheap even for Kerr (unlike Riemann products), so compute it
        on demand; None if it fails."""
        if not hasattr(self, "_ginv_sym"):
            try:
                self._ginv_sym = self.g.inv()
            except Exception:  # noqa: BLE001
                self._ginv_sym = None
        return self._ginv_sym

    # -- horizon: roots of g^rr (event horizon), also g_tt (static) --------- #
    def detect_horizon(self) -> dict:
        coords = self.spec.coords
        t_idx = _find_role(coords, self.spec.coord_roles, "time")
        r_idx = _find_role(coords, self.spec.coord_roles, "radial")
        if r_idx is None:
            return {"has_horizon": False, "note": "need a radial coord"}
        r = self.coord_syms[r_idx]
        subs = {s: sp.Integer(1) for s in self.param_syms.values()}

        roots_grr, roots_gtt = [], []
        try:  # event horizon: null hypersurface where g^{rr} = 0
            grr_up = self._symbolic_ginv()[r_idx, r_idx].subs(subs)
            roots_grr = [rt for rt in sp.solve(sp.Eq(grr_up, 0), r) if _is_real_positive(rt)]
        except Exception:  # noqa: BLE001
            pass
        if t_idx is not None:  # Killing/ergo surface (static)
            try:
                gtt = self.g[t_idx, t_idx].subs(subs)
                roots_gtt = [rt for rt in sp.solve(sp.Eq(gtt, 0), r) if _is_real_positive(rt)]
            except Exception:  # noqa: BLE001
                pass
        has = len(roots_grr) > 0 or len(roots_gtt) > 0
        return {
            "has_horizon": has,
            "g_rr_upper_roots_at_unit_params": [str(x) for x in roots_grr],
            "g_tt_roots_at_unit_params": [str(x) for x in roots_gtt],
        }

    # -- energy conditions (numeric, orthonormal frame) --------------------- #
    def energy_conditions(self, points) -> dict:
        results = {"null": True, "weak": True, "strong": True, "dominant": True}
        checked = 0
        for pt in points:
            cur = self._curvature_at(pt)
            if cur is None:
                continue
            g, Tl = cur["g"], cur["T"]
            rp = _rho_pressures(g, Tl)
            if rp is None:
                results["null"] = results["weak"] = False
                results["strong"] = results["dominant"] = False
                results["indeterminate"] = True
                continue
            rho, ps = rp
            checked += 1
            nec = all(rho + p >= -1e-8 for p in ps)
            wec = nec and rho >= -1e-8
            sec = nec and (rho + sum(ps) >= -1e-8)
            dec = rho >= -1e-8 and all(rho + 1e-8 >= abs(p) for p in ps)
            results["null"] &= nec
            results["weak"] &= wec
            results["strong"] &= sec
            results["dominant"] &= dec
        results["n_checked"] = checked
        if checked == 0:
            results["null"] = results["weak"] = False
            results["strong"] = results["dominant"] = False
        return results


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _infer_role(name: str) -> str:
    n = name.lower()
    if n in ("t", "time", "x0", "tau", "v", "u"):
        return "time"
    if n in ("r", "rho", "radius", "x1"):
        return "radial"
    if n in ("theta", "θ", "th"):
        return "angle"
    if n in ("phi", "φ", "varphi", "psi", "azimuth"):
        return "azimuth"
    return "generic"


def _find_role(coords, roles, target) -> Optional[int]:
    for i, name in enumerate(coords):
        if (roles.get(name) or _infer_role(name)) == target:
            return i
    return None


def _mode(items):
    if not items:
        return None
    counts: dict = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _is_real_positive(expr) -> bool:
    try:
        v = complex(expr)
        return abs(v.imag) < 1e-9 and v.real > 0
    except (TypeError, ValueError):
        return False


def _rho_pressures(g: np.ndarray, Tl: np.ndarray):
    """Return (energy_density rho, [principal_pressures]) in an orthonormal frame
    built from the METRIC, or None if T is not a real type-I tensor.

    Building the frame from the metric (not from T's eigenvectors) is robust to
    isotropic stress tensors T ∝ g (e.g. a cosmological-constant fluid), whose
    degenerate spectrum makes eigenvector classification ambiguous.

    Frame: eigendecompose g = V diag(w) V^T; e_i = V_i / sqrt|w_i| gives an
    orthonormal basis (e_i·e_j = η_ij). With u = e_0 the unit timelike leg,
    rho = T_ab u^a u^b and p_i = T_ab e_i^a e_i^b (mostly-plus)."""
    n = g.shape[0]
    try:
        w, V = np.linalg.eigh(0.5 * (g + g.T))
    except np.linalg.LinAlgError:
        return None
    neg = [i for i in range(n) if w[i] < 0]
    pos = [i for i in range(n) if w[i] > 0]
    if len(neg) != 1 or len(pos) != n - 1:
        return None
    E = np.zeros((n, n))  # columns: g-orthonormal basis vectors
    for i in range(n):
        E[:, i] = V[:, i] / math.sqrt(abs(w[i]))
    That = E.T @ Tl @ E  # T in the orthonormal frame
    # off-diagonal in the orthonormal frame => not Hawking-Ellis type I
    diag_scale = max(float(np.max(np.abs(np.diag(That)))), 1.0)
    off = That - np.diag(np.diag(That))
    if float(np.max(np.abs(off))) > 1e-6 * diag_scale:
        return None
    ti = neg[0]
    rho = float(That[ti, ti])          # T_{\hat0\hat0} = energy density
    ps = [float(That[k, k]) for k in pos]
    return rho, ps
