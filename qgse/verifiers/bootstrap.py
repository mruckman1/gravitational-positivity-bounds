"""Conformal-bootstrap verifier (single-correlator gap bound) — Phase 3.

This is the first *discriminating* verifier: unlike vacuum GR (all necessity,
answer already known), here "feasible / excluded" is a fact about CFT space that
nobody handed the model. Via AdS/CFT a consistent unitary CFT *defines* a
quantum-gravity theory in the bulk, so a genuine crossing-symmetry constraint is
the closest computable thing to a real quantum-gravity consistency condition.

A candidate is a spectrum ASSUMPTION for a single scalar 4-point function
<phi phi phi phi> in d dimensions: external dimension Δ_φ and an assumed gap Δ*
above the identity in the spin-0 channel. The verifier asks SDPB (the exact
Rattazzi–Rychkov linear-functional method) whether crossing symmetry + unitarity
admit such a CFT:

  * feasible  -> the assumption is ALLOWED (valid=True),
  * excluded  -> SDPB found a dual functional PROVING no such CFT exists; that
                 functional is the auditable certificate.

The maximal feasible Δ* as a function of Δ_φ is the gap-bound curve whose kink is
the 3D Ising model — the known-answer sanity check.

Pipeline: PyCFTBoot (pure-Python block/SDP generator, vendored) -> SDPB (run in
the wlandry/sdpb Docker container via shims in vendor_sdpb_bin/). Nothing here
consults an LLM.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from qgse.interfaces import Candidate, Verifier, VerdictGrounded

_REPO = Path(__file__).resolve().parents[2]
_PYCFTBOOT_DIR = _REPO / "vendor_pycftboot"
_SHIM_DIR = _REPO / "vendor_sdpb_bin"

# In-process cache of (Convolved) block tables — the expensive, Δ_φ-independent
# part. Keyed by (dim, k_max, l_max, m_max, n_max).
_TABLE_CACHE: dict = {}


_PCB = None


def _ensure_env() -> None:
    shim = str(_SHIM_DIR)
    if shim not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = shim + os.pathsep + os.environ.get("PATH", "")
    if str(_PYCFTBOOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PYCFTBOOT_DIR))
    # Prefer the NATIVE arm64 SDPB image on Apple Silicon (fast: ~2s/solve vs
    # ~10s emulated). Falls back to the shim default (amd64 wlandry, emulated) on
    # other hosts. Both are overridable via QGSE_SDPB_IMAGE / QGSE_SDPB_PLATFORM.
    mach = platform.machine().lower()
    if mach in ("arm64", "aarch64"):
        if not os.environ.get("QGSE_SDPB_IMAGE"):
            os.environ["QGSE_SDPB_IMAGE"] = "bootstrapcollaboration/sdpb:3.0.0"
        if "bootstrapcollaboration" in os.environ.get("QGSE_SDPB_IMAGE", ""):
            os.environ.setdefault("QGSE_SDPB_PLATFORM", "linux/arm64")
    # Consistent multi-proc: SDPB 3.0 hits an Elemental block-distribution
    # assertion at higher derivative order under single-rank; converting AND
    # solving with the same >=2 proc count clears it (verified n_max=4). Harmless
    # at low order. Overridable via QGSE_SDPB_NPROC.
    os.environ.setdefault("QGSE_SDPB_NPROC", "4")


def _import_pcb():
    """Import PyCFTBoot. Its bootstrap.py does exec(open("common.py")) etc. at
    import time, reading sibling files relative to CWD — so the import MUST run
    with CWD = the PyCFTBoot directory. Cached thereafter."""
    global _PCB
    if _PCB is None:
        _ensure_env()
        prev = os.getcwd()
        os.chdir(_PYCFTBOOT_DIR)
        try:
            import bootstrap as pcb
            _PCB = pcb
        finally:
            os.chdir(prev)
    return _PCB


@contextlib.contextmanager
def _in_dir(path: Path):
    prev = os.getcwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@dataclass
class BootstrapSpec:
    """A single-correlator spectrum assumption to test for crossing consistency.

    dim:         spacetime dimension d (e.g. 3.0).
    delta_phi:   external scalar dimension Δ_φ.
    scalar_gap:  assumed gap Δ* in the spin-0 (scalar) channel above the identity.
    k_max/l_max/m_max/n_max: PyCFTBoot block-table truncation / derivative order.
                 Higher = sharper bound but much slower (SDPB runs emulated here).
    spin_channel: which spin's gap is assumed (0 = scalar).
    """

    dim: float = 3.0
    delta_phi: float = 0.518
    scalar_gap: float = 1.0
    k_max: int = 20
    l_max: int = 15
    m_max: int = 2
    n_max: int = 4
    spin_channel: int = 0

    @staticmethod
    def from_dict(d: dict) -> "BootstrapSpec":
        return BootstrapSpec(
            dim=float(d.get("dim", 3.0)),
            delta_phi=float(d["delta_phi"]),
            scalar_gap=float(d.get("scalar_gap", 1.0)),
            k_max=int(d.get("k_max", 20)),
            l_max=int(d.get("l_max", 15)),
            m_max=int(d.get("m_max", 2)),
            n_max=int(d.get("n_max", 4)),
            spin_channel=int(d.get("spin_channel", 0)),
        )

    def table_key(self):
        return (self.dim, self.k_max, self.l_max, self.m_max, self.n_max)


class BootstrapVerifier(Verifier):
    domain = "bootstrap"

    def __init__(self, *, work_dir: Optional[str] = None, procs_per_node: int = 1,
                 admissibility_gate=None) -> None:
        self.work_dir = Path(work_dir) if work_dir else (_REPO / "results" / "bootstrap_work")
        self.procs_per_node = procs_per_node
        # §5b consumer: if set, discovered/proven constraints pre-filter a spectrum
        # BEFORE the (expensive) SDPB solve — the dashed loop closing on this
        # verifier. A pre-filtered spectrum returns an inadmissible verdict at ~0
        # cost, with no solve. Optional; None preserves stock behavior.
        self.admissibility_gate = admissibility_gate

    def known_limits(self) -> list[str]:
        # Admissibility for a CFT spectrum assumption.
        return ["crossing_symmetry", "unitarity", "identity_normalized"]

    # -- block tables (cached) ---------------------------------------------- #
    def _tables(self, spec: BootstrapSpec):
        key = spec.table_key()
        if key not in _TABLE_CACHE:
            pcb = _import_pcb()
            t1 = pcb.ConformalBlockTable(spec.dim, spec.k_max, spec.l_max,
                                         spec.m_max, spec.n_max)
            t2 = pcb.ConvolvedBlockTable(t1)
            _TABLE_CACHE[key] = t2
        return _TABLE_CACHE[key]

    def _make_sdp(self, spec: BootstrapSpec):
        pcb = _import_pcb()
        t2 = self._tables(spec)
        sdp = pcb.SDP(spec.delta_phi, t2)
        try:
            sdp.set_option("procsPerNode", str(self.procs_per_node))
        except Exception:  # noqa: BLE001
            pass
        return sdp

    # -- public entry points ------------------------------------------------ #
    def verify(self, c: Candidate) -> VerdictGrounded:
        try:
            if isinstance(c.meta.get("bootstrap_spec"), dict):
                spec = BootstrapSpec.from_dict(c.meta["bootstrap_spec"])
            else:
                spec = BootstrapSpec.from_dict(json.loads(c.artifact))
        except Exception as e:  # noqa: BLE001
            return self._error(f"could not parse bootstrap spec: {e!r}")
        return self.verify_spec(spec)

    def _spec_dirname(self, spec: BootstrapSpec) -> str:
        # Readable prefix + hash of ALL fields at full precision — formatting
        # truncation must never let two distinct problems share a solve dir.
        import hashlib
        from dataclasses import asdict
        blob = json.dumps(asdict(spec), sort_keys=True, default=str)
        h = hashlib.sha1(blob.encode()).hexdigest()[:10]
        return f"pt_dphi{spec.delta_phi:.6f}_gap{spec.scalar_gap:.6f}_{h}"

    def _prefiltered(self, spec: BootstrapSpec, violations: list, cost: float) -> VerdictGrounded:
        """A spectrum ruled inadmissible by a discovered constraint, WITHOUT a
        solve. Not an error (the pipeline is fine) — a legitimate exclusion by the
        §5b loop, carrying which constraints it violated."""
        return VerdictGrounded(
            valid=False, fitness=-1.0, certificate=None,
            reproduces_known={"spectrum_allowed": False}, cost=cost,
            details={"feasible": False, "prefiltered": True,
                     "delta_phi": spec.delta_phi, "scalar_gap": spec.scalar_gap,
                     "dim": spec.dim, "violations": violations,
                     "terminateReason": "prefiltered by discovered constraint"})

    def verify_spec(self, spec: BootstrapSpec) -> VerdictGrounded:
        _ensure_env()
        t0 = time.time()
        # §5b pre-filter: reject a spectrum that a discovered/proven constraint
        # already rules out, before paying for SDPB. Screens the spec's own
        # parameter dict (delta_phi / scalar_gap / dim), which is exactly what a
        # compiled constraint's field_path targets.
        if self.admissibility_gate is not None:
            from dataclasses import asdict as _asdict
            report = self.admissibility_gate.screen(_asdict(spec))
            if not report.admissible:
                return self._prefiltered(spec, report.violations, time.time() - t0)
        try:
            # Tables are built/cached in-process (CWD-independent); only the
            # solve runs inside a FRESH per-spec directory. Checkpoint hygiene:
            # SDPB auto-loads a checkpoint matching its default names, so a
            # shared directory lets one solve inherit another problem's (or
            # another rank-count's) state — wiped per-spec dirs prevent that.
            sdp = self._make_sdp(spec)
            spec_dir = self.work_dir / self._spec_dirname(spec)
            import shutil as _shutil
            if spec_dir.exists():
                _shutil.rmtree(spec_dir)
            with _in_dir(spec_dir):
                sdp.set_bound(spec.spin_channel, float(spec.scalar_gap))
                name = "qgse_bootstrap"
                sdp.iterate(name=name)  # verdict read from terminateReason below
                output = self._safe_read_output(sdp, name)
                from qgse.verifiers.bootstrap_mixed import _cleanup_solver_files
                _cleanup_solver_files(name)
        except Exception as e:  # noqa: BLE001
            return self._error(f"bootstrap pipeline failed: {e!r}", cost=time.time() - t0)

        cost = time.time() - t0
        terminate = str(output.get("terminateReason", "unknown"))
        # Three-way classification — ONLY these terminations carry a certificate:
        #   "found primal feasible solution" -> spectrum ALLOWED
        #   "found dual feasible solution"   -> EXCLUDED (dual functional proof)
        # Anything else (maxIterations/maxRuntime/maxComplementarity exceeded)
        # means the solver STALLED: treating it as excluded would fake a
        # certificate and silently bias gap_bound low. Inconclusive -> error.
        if terminate == "found primal feasible solution":
            feasible = True
        elif terminate == "found dual feasible solution":
            feasible = False
        else:
            return self._error(
                f"SDPB inconclusive: terminateReason={terminate!r} — no "
                f"feasibility certificate; raise maxIterations/maxRuntime and re-run",
                cost=cost,
            )
        certificate = json.dumps({
            "kind": "sdpb_single_correlator_gap",
            "dim": spec.dim,
            "delta_phi": spec.delta_phi,
            "scalar_gap": spec.scalar_gap,
            "spin_channel": spec.spin_channel,
            "orders": {"k_max": spec.k_max, "l_max": spec.l_max,
                       "m_max": spec.m_max, "n_max": spec.n_max},
            "feasible": feasible,
            "terminateReason": terminate,
            "primalObjective": output.get("primalObjective"),
            "dualObjective": output.get("dualObjective"),
        }, sort_keys=True)

        reproduces = {
            "crossing_symmetry": True,     # SDP is built FROM the crossing equation
            "unitarity": True,             # OPE-coefficient positivity is imposed
            "identity_normalized": True,   # identity contributes the inhomogeneous term
            "spectrum_allowed": feasible,
        }
        # Grounded fitness: reward a large ALLOWED gap (climbs toward the bound);
        # any excluded assumption scores below every feasible one.
        fitness = float(spec.scalar_gap) if feasible else -1.0

        verdict = VerdictGrounded(
            valid=feasible,
            fitness=fitness,
            certificate=certificate,
            reproduces_known=reproduces,
            cost=cost,
            details={
                "feasible": feasible,
                "terminateReason": terminate,
                "delta_phi": spec.delta_phi,
                "scalar_gap": spec.scalar_gap,
                "dim": spec.dim,
            },
        )
        verdict.audit()
        return verdict

    def gap_bound(self, spec: BootstrapSpec, lower: float, upper: float,
                  threshold: float = 0.01) -> dict:
        """Bisect to the maximal allowed spin-0 gap at this Δ_φ (the gap-bound
        curve; its kink is the 3D Ising point). Returns the bound + timing.

        Manual bisection through :meth:`verify_spec` so every step solves in its
        own fresh directory (PyCFTBoot's ``sdp.bisect`` reuses one name across
        steps, letting SDPB warm-start from the previous step's checkpoint —
        sound at fixed rank count, but we prefer fully independent verdicts).
        Semantics match ``sdp.bisect``: returns the allowed value closest to the
        boundary. Tables are cached in-process, so per-step cost is solve-only."""
        _ensure_env()
        t0 = time.time()
        from dataclasses import replace as _replace
        lo, hi = float(lower), float(upper)   # lo assumed allowed, hi disallowed
        while hi - lo > threshold:
            mid = 0.5 * (lo + hi)
            v = self.verify_spec(_replace(spec, scalar_gap=mid))
            if v.error:
                raise RuntimeError(f"bisection step failed at gap={mid}: {v.error}")
            if v.valid:
                lo = mid
            else:
                hi = mid
        bound = lo
        return {"delta_phi": spec.delta_phi, "dim": spec.dim,
                "gap_bound": float(bound), "cost_s": time.time() - t0,
                "orders": spec.table_key()}

    # -- internals ---------------------------------------------------------- #
    def _safe_read_output(self, sdp, name: str) -> dict:
        try:
            raw = sdp.read_output(name=name)
        except Exception:  # noqa: BLE001
            return {}
        # SDPB objectives come back as arbitrary-precision RealMPFR objects;
        # coerce everything to JSON-safe types (keeping full precision as strings).
        out = {}
        for k, v in raw.items():
            out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
        return out

    def _error(self, msg: str, cost: float = 0.0) -> VerdictGrounded:
        return VerdictGrounded(valid=False, fitness=-1e9, certificate=None,
                               reproduces_known={}, cost=cost, error=msg)
