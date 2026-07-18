"""O(N) vector-model conformal bootstrap (single correlator) — un-mapped territory.

Everything else in this repo is Z2/Ising (`bootstrap.py`, `bootstrap_mixed.py`).
This adds the first NON-Z2 global symmetry: the O(N) vector model, whose
⟨φ_i φ_j φ_k φ_l⟩ four-point function decomposes the φ×φ OPE into three irreps —
**singlet** (S, even spin), **symmetric-traceless** (T, even spin), and
**antisymmetric** (A, odd spin) — giving a *three-component* vectorial sum rule
instead of the single scalar equation. The maximal allowed gap on the leading
singlet scalar, as a function of Δ_φ, has a **kink at the O(N) vector model**
(the analog of the Ising kink): O(2) is the 3D XY universality class
(superfluid ⁴He), O(3) the Heisenberg (isotropic magnets), N→∞ the free/critical
sphere model. This is a real quantum-gravity-adjacent object: via AdS/CFT each
consistent O(N) CFT defines a bulk theory with an O(N) global symmetry.

Faithful transcription of PyCFTBoot ``tutorial.py`` choice 2 (the O(N) singlet
bound); the crossing coefficients (with the characteristic ``1 ∓ 2/N`` and
``1 + 2/N`` factors) come from the O(N) 6j / projector algebra. Pipeline and
soundness discipline are identical to :mod:`qgse.verifiers.bootstrap`:
PyCFTBoot → SDPB (native-arm64 Docker), 3-way terminateReason classification
(feasible / excluded-with-dual-proof / inconclusive→error, never a fake
certificate). The block generator and SDP layer are unchanged — only the crossing
``vector_types`` differ from the Z2 case, so O(N) at any dimension the generator
supports (odd / non-integer d) is immediately reachable.

Reference: Kos, Poland, Simmons-Duffin, "Bootstrapping the O(N) vector models",
JHEP 06 (2014) 091 [arXiv:1307.6856].
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from qgse.interfaces import Candidate, Verifier, VerdictGrounded
from qgse.verifiers.bootstrap import _ensure_env, _import_pcb, _in_dir, _REPO

# In-process cache of the two convolved O(N) tables. Like the single-correlator
# Z2 case they depend only on (dim, k_max, l_max, m_max, n_max) — NOT on N or
# Δ_φ (N enters only the crossing coefficients, Δ_φ only the SDP prefactor), so
# one table build serves an entire (N, Δ_φ, gap) scan at fixed order/dimension.
_ON_TABLE_CACHE: dict = {}


def _on_info(pcb, tab_sym, tab_asym, N: float):
    """The O(N) three-channel vectorial sum rule (tutorial.py choice 2).

    tab_list = [tab_sym, tab_asym] where tab_sym = ConvolvedBlockTable(g, symmetric=True)
    (the H convolution) and tab_asym = ConvolvedBlockTable(g) (the F convolution).
    Each vector component is a [coefficient, table-index] pair; index 0 -> tab_sym,
    index 1 -> tab_asym. Spins: singlet/symmetric even, antisymmetric odd.
    """
    vec1 = [[0, 1], [1, 1], [1, 0]]                                   # singlet
    vec2 = [[1, 1], [1.0 - (2.0 / N), 1], [-(1.0 + (2.0 / N)), 0]]     # symmetric
    vec3 = [[1, 1], [-1, 1], [1, 0]]                                  # antisymmetric
    return [[vec1, 0, "singlet"], [vec2, 0, "symmetric"], [vec3, 1, "antisymmetric"]]


@dataclass
class ONSpec:
    """An O(N) spectrum assumption: a gap on the leading SINGLET scalar in the
    φ×φ OPE, at external dimension Δ_φ, N-component vector, in d dimensions.

    dim:         spacetime dimension d.
    delta_phi:   external vector-scalar dimension Δ_φ.
    N:           number of vector components (O(N) symmetry). Float so N→large and
                 fractional N (analytic continuation) are reachable.
    singlet_gap: assumed gap on the leading singlet scalar above the identity.
    channel:     which irrep's scalar gap is assumed ("singlet" by default).
    """

    dim: float = 3.0
    delta_phi: float = 0.519
    N: float = 2.0
    singlet_gap: float = 1.0
    channel: str = "singlet"
    k_max: int = 20
    l_max: int = 20
    # m_max=1 is REQUIRED for this single-correlator O(N) system, not a perf knob:
    # for the equal-external correlator the crossing function F is antisymmetric
    # under z↔1-z, so only odd-m derivatives survive at the crossing-symmetric
    # point. PyCFTBoot's m_max=1 selects exactly that parity basis; m_max=2 mixes
    # in even-m derivatives, giving a NON-NESTED, strictly weaker functional space
    # whose bound is spuriously LOOSE (verified: at m_max=2 the O(3) singlet bound
    # exceeds 2.2, vs the PyCFTBoot reference ~1.7 at m_max=1). PyCFTBoot's own
    # tutorial uses m_max=1 for O(N) (choice 2) for this reason. Note the failure
    # is in the SAFE direction — it only fails to exclude, never fakes an exclusion.
    m_max: int = 1
    n_max: int = 4

    @staticmethod
    def from_dict(d: dict) -> "ONSpec":
        return ONSpec(
            dim=float(d.get("dim", 3.0)),
            delta_phi=float(d["delta_phi"]),
            N=float(d.get("N", 2.0)),
            singlet_gap=float(d.get("singlet_gap", 1.0)),
            channel=str(d.get("channel", "singlet")),
            k_max=int(d.get("k_max", 20)),
            l_max=int(d.get("l_max", 20)),
            m_max=int(d.get("m_max", 2)),
            n_max=int(d.get("n_max", 4)),
        )

    def table_key(self):
        return (self.dim, self.k_max, self.l_max, self.m_max, self.n_max)


class ONVerifier(Verifier):
    domain = "bootstrap_on"

    def __init__(self, *, work_dir: Optional[str] = None) -> None:
        self.work_dir = Path(work_dir) if work_dir else (_REPO / "results" / "on_work")

    def known_limits(self) -> list[str]:
        return ["crossing_symmetry", "unitarity", "on_global_symmetry"]

    # -- block tables (cached; the dominant cost) --------------------------- #
    def _tables(self, spec: ONSpec):
        key = spec.table_key()
        if key not in _ON_TABLE_CACHE:
            pcb = _import_pcb()
            # odd_spins=True is MANDATORY: O(N) has an antisymmetric (odd-spin)
            # channel, unlike the Z2 single-correlator case.
            g = pcb.ConformalBlockTable(spec.dim, spec.k_max, spec.l_max,
                                        spec.m_max, spec.n_max, odd_spins=True)
            tab_sym = pcb.ConvolvedBlockTable(g, symmetric=True)
            tab_asym = pcb.ConvolvedBlockTable(g)
            _ON_TABLE_CACHE[key] = (tab_sym, tab_asym)
        return _ON_TABLE_CACHE[key]

    # -- public entry points ------------------------------------------------ #
    def verify(self, c: Candidate) -> VerdictGrounded:
        try:
            if isinstance(c.meta.get("on_spec"), dict):
                spec = ONSpec.from_dict(c.meta["on_spec"])
            else:
                spec = ONSpec.from_dict(json.loads(c.artifact))
        except Exception as e:  # noqa: BLE001
            return self._error(f"could not parse O(N) spec: {e!r}")
        return self.verify_spec(spec)

    def _spec_dirname(self, spec: ONSpec) -> str:
        import hashlib
        blob = json.dumps(asdict(spec), sort_keys=True, default=str)
        h = hashlib.sha1(blob.encode()).hexdigest()[:10]
        return f"on_N{spec.N:g}_dphi{spec.delta_phi:.6f}_gap{spec.singlet_gap:.6f}_{h}"

    def verify_spec(self, spec: ONSpec) -> VerdictGrounded:
        _ensure_env()
        pcb = _import_pcb()
        pcb.cutoff = 0  # exact poles (pinned: approximate poles unacceptable)
        t0 = time.time()
        try:
            tab_sym, tab_asym = self._tables(spec)
            sdp = pcb.SDP(spec.delta_phi, [tab_sym, tab_asym],
                          vector_types=_on_info(pcb, tab_sym, tab_asym, spec.N))
            # Assume a gap on the leading scalar in the chosen channel; every
            # other channel keeps its unitarity-bound default. The maximal
            # feasible singlet gap vs Δ_φ is the O(N) bound curve (kink = O(N) model).
            sdp.set_bound([0, spec.channel], float(spec.singlet_gap))
            spec_dir = self.work_dir / self._spec_dirname(spec)
            if spec_dir.exists():
                shutil.rmtree(spec_dir)
            with _in_dir(spec_dir):
                name = "qgse_on"
                sdp.iterate(name=name)  # verdict read from terminateReason below
                out = self._safe_read_output(sdp, name)
                from qgse.verifiers.bootstrap_mixed import _cleanup_solver_files
                _cleanup_solver_files(name)
        except Exception as e:  # noqa: BLE001
            return self._error(f"O(N) pipeline failed: {e!r}", cost=time.time() - t0)

        cost = time.time() - t0
        terminate = str(out.get("terminateReason", "unknown"))
        # Three-way classification, identical discipline to bootstrap.py: only a
        # primal- or dual-feasible termination carries a certificate; a stall
        # (maxIterations/maxRuntime/...) is inconclusive -> error, NEVER a fake
        # exclusion.
        if terminate == "found primal feasible solution":
            feasible = True
        elif terminate == "found dual feasible solution":
            feasible = False
        else:
            return self._error(
                f"SDPB inconclusive: terminateReason={terminate!r} — no feasibility "
                f"certificate; raise maxIterations/maxRuntime and re-run", cost=cost)

        certificate = json.dumps({
            "kind": "sdpb_on_singlet_bound",
            "global_symmetry": f"O({spec.N:g})",
            "dim": spec.dim, "delta_phi": spec.delta_phi, "N": spec.N,
            "channel": spec.channel, "singlet_gap": spec.singlet_gap,
            "orders": {"k_max": spec.k_max, "l_max": spec.l_max,
                       "m_max": spec.m_max, "n_max": spec.n_max},
            "feasible": feasible, "terminateReason": terminate,
            "primalObjective": out.get("primalObjective"),
            "dualObjective": out.get("dualObjective"),
        }, sort_keys=True)
        reproduces = {
            "crossing_symmetry": True, "unitarity": True,
            "on_global_symmetry": True, "spectrum_allowed": feasible,
        }
        fitness = float(spec.singlet_gap) if feasible else -1.0
        verdict = VerdictGrounded(
            valid=feasible, fitness=fitness, certificate=certificate,
            reproduces_known=reproduces, cost=cost,
            details={"feasible": feasible, "terminateReason": terminate,
                     "delta_phi": spec.delta_phi, "N": spec.N,
                     "singlet_gap": spec.singlet_gap, "dim": spec.dim,
                     "channel": spec.channel})
        verdict.audit()
        return verdict

    def gap_bound(self, spec: ONSpec, lower: float, upper: float,
                  threshold: float = 0.02) -> dict:
        """Bisect to the maximal allowed gap on the leading channel scalar at this
        (N, Δ_φ). The bound curve's kink locates the O(N) vector model. Manual
        bisection through :meth:`verify_spec` so each step solves in its own fresh
        directory (independent verdicts); tables are cached, so per-step cost is
        solve-only."""
        _ensure_env()
        t0 = time.time()
        from dataclasses import replace as _replace
        lo, hi = float(lower), float(upper)   # lo assumed allowed, hi disallowed
        while hi - lo > threshold:
            mid = 0.5 * (lo + hi)
            v = self.verify_spec(_replace(spec, singlet_gap=mid))
            if v.error:
                raise RuntimeError(f"bisection step failed at gap={mid}: {v.error}")
            (lo, hi) = (mid, hi) if v.valid else (lo, mid)
        return {"N": spec.N, "delta_phi": spec.delta_phi, "dim": spec.dim,
                "channel": spec.channel, "gap_bound": float(lo),
                "cost_s": time.time() - t0, "orders": spec.table_key()}

    # -- internals ---------------------------------------------------------- #
    def _safe_read_output(self, sdp, name: str) -> dict:
        try:
            raw = sdp.read_output(name=name)
        except Exception:  # noqa: BLE001 — never fabricate a feasibility claim
            return {}
        out = {}
        for k, v in raw.items():
            out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
        return out

    def _error(self, msg: str, cost: float = 0.0) -> VerdictGrounded:
        return VerdictGrounded(valid=False, fitness=-1e9, certificate=None,
                               reproduces_known={}, cost=cost, error=msg)
