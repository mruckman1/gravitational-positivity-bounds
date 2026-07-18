"""Mixed-correlator conformal bootstrap (σ–ε system) — the 3D Ising *island*.

The single-correlator gap bound is one number per Δ_φ. The mixed-correlator
system ⟨σσσσ⟩, ⟨σσεε⟩, ⟨εεεε⟩ with Z2 symmetry is far more constraining: under
the assumption of exactly ONE relevant Z2-odd scalar (σ) and ONE relevant
Z2-even scalar (ε), crossing + unitarity carve out a small *closed* allowed
region in the (Δ_σ, Δ_ε) plane — the Kos–Poland–Simmons-Duffin **island** that
pins the 3D Ising CFT. A point OUTSIDE the island is a genuine EXCLUSION with an
SDPB dual-functional proof.

This is the discriminating object worth scanning: whether a proposed (Δ_σ, Δ_ε)
survives is a computed fact, and the boundary of the island in less-studied
regions is where new discrimination could live.

Faithful transcription of PyCFTBoot's `tutorial.py` choice 3. Requires
RESEARCH-ORDER SDPB (n_max≈4) — the island does not close at low order — and the
consistent-multi-proc SDPB setting (QGSE_SDPB_NPROC≥2) that clears the SDPB-3.0
single-proc block assertion. Pipeline as in :mod:`qgse.verifiers.bootstrap`.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from qgse.interfaces import Candidate, Verifier, VerdictGrounded
from qgse.verifiers.bootstrap import _ensure_env, _import_pcb, _in_dir, _REPO

# --------------------------------------------------------------------------- #
# In-process block-table cache — the dominant cost of an island point.
#
# Key physics fact exploited here: of the three block tables, g1 depends only on
# (dim, order) — NOT on the external dimensions at all — and g2/g3 depend on the
# externals only through the DIFFERENCE d12 = Δ_ε − Δ_σ. So a scan parametrized
# as (Δ_σ, d12) reuses every table across a whole constant-d12 slice, reducing
# table generation from per-point to per-slice. The convolved tables (what the
# SDP actually consumes) are what we cache.
#
# Structure: {(dim,k,l,m,n): {"f1": (f1a, f1s), "d12": OrderedDict{d12: (f2a, f2s, f3)}}}
# The d12 sub-cache is LRU-bounded — tables are large symengine objects.
# --------------------------------------------------------------------------- #
_MIXED_TABLE_CACHE: dict = {}
_D12_CACHE_MAX = 4


def _cleanup_solver_files(name: str) -> None:
    """Delete the regenerable bulk a solve leaves in CWD (converted SDP blocks,
    binary checkpoints, the problem XML — ~100 MB/point at research order),
    keeping ``<name>_out/`` — the SDPB output holding the dual functional, i.e.
    the auditable certificate artifact. A multi-hour campaign must not eat the
    disk. Failures are swallowed: cleanup must never affect a verdict."""
    import shutil as _sh
    for victim in (name, f"{name}.ck", f"{name}.ck.bk", f"{name}.xml"):
        try:
            if os.path.isdir(victim):
                _sh.rmtree(victim)
            elif os.path.exists(victim):
                os.remove(victim)
        except OSError:
            pass


# --- Fixed Z2 crossing structure (independent of the specific dimensions) ---
# Quads are [coefficient, convolved-table-index, dim-index-1, dim-index-2] over
# tab_list = [f_tab1a, f_tab1s, f_tab2a, f_tab2s, f_tab3] and dim_list=[Δ_σ, Δ_ε].
def _z2_info():
    vec3 = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 0], [-1, 2, 0, 0], [1, 3, 0, 0]]
    vec2 = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 0], [1, 2, 0, 0], [-1, 3, 0, 0]]
    m1 = [[[1, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]]
    m2 = [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [1, 0, 1, 1]]]
    m3 = [[[0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]]
    m4 = [[[0, 0, 0, 0], [0.5, 0, 0, 1]], [[0.5, 0, 0, 1], [0, 0, 0, 0]]]
    m5 = [[[0, 1, 0, 0], [0.5, 1, 0, 1]], [[0.5, 1, 0, 1], [0, 1, 0, 0]]]
    vec1 = [m1, m2, m3, m4, m5]
    return [[vec1, 0, "z2-even-l-even"],
            [vec2, 0, "z2-odd-l-even"],
            [vec3, 1, "z2-odd-l-odd"]]


@dataclass
class IslandPoint:
    """A candidate 3D-Ising-type point to test for membership in the allowed
    island (single relevant Z2-odd + single relevant Z2-even scalar)."""

    delta_sigma: float          # Z2-odd external scalar dimension
    delta_epsilon: float        # Z2-even external scalar dimension
    dim: float = 3.0
    k_max: int = 20
    l_max: int = 20
    m_max: int = 2
    n_max: int = 4              # research order — the island needs it
    dual_error_threshold: str = "1e-15"

    @staticmethod
    def from_dict(d: dict) -> "IslandPoint":
        return IslandPoint(
            delta_sigma=float(d["delta_sigma"]),
            delta_epsilon=float(d["delta_epsilon"]),
            dim=float(d.get("dim", 3.0)),
            k_max=int(d.get("k_max", 20)),
            l_max=int(d.get("l_max", 20)),
            m_max=int(d.get("m_max", 2)),
            n_max=int(d.get("n_max", 4)),
            dual_error_threshold=str(d.get("dual_error_threshold", "1e-15")),
        )


class MixedCorrelatorVerifier(Verifier):
    domain = "bootstrap_mixed"

    def __init__(self, *, work_dir: Optional[str] = None) -> None:
        self.work_dir = Path(work_dir) if work_dir else (_REPO / "results" / "island_work")

    def known_limits(self) -> list[str]:
        return ["crossing_symmetry", "unitarity", "z2_symmetry",
                "single_relevant_z2_odd", "single_relevant_z2_even"]

    def verify(self, c: Candidate) -> VerdictGrounded:
        try:
            if isinstance(c.meta.get("island_point"), dict):
                pt = IslandPoint.from_dict(c.meta["island_point"])
            else:
                pt = IslandPoint.from_dict(json.loads(c.artifact))
        except Exception as e:  # noqa: BLE001
            return self._error(f"could not parse island point: {e!r}")
        return self.verify_point(pt)

    def verify_point(self, pt: IslandPoint) -> VerdictGrounded:
        _ensure_env()
        pcb = _import_pcb()
        pcb.cutoff = 0  # exact poles (PyCFTBoot default, pinned: approximate
        #                 poles are not acceptable for island verdicts)
        t0 = time.time()
        try:
            # Tables are pure in-memory computation (no CWD dependence) and are
            # cached across points — build them before entering the solve dir.
            tabs = self._tables(pcb, pt)
            # Checkpoint hygiene: a FRESH, wiped directory per point, so no solve
            # can inherit another problem's (or another rank-count's) SDPB
            # checkpoint via the shared default names.
            point_dir = self.work_dir / self._point_dirname(pt)
            if point_dir.exists():
                shutil.rmtree(point_dir)
            with _in_dir(point_dir):
                terminate = self._solve(pcb, pt, tabs)
        except Exception as e:  # noqa: BLE001
            return self._error(f"mixed-correlator pipeline failed: {e!r}",
                               cost=time.time() - t0)

        # Three-way verdict classification. ONLY these two terminations carry a
        # feasibility certificate; anything else (maxIterations/maxRuntime/
        # maxComplementarity exceeded, ...) means the solver STALLED — recording
        # it as an exclusion would be a fake certified result, the one thing
        # this project must never emit. Inconclusive -> error verdict (no
        # certificate), so scans mark it '?' and retries/parameter bumps are
        # visible instead of silently corrupting the island map.
        if terminate == "found primal feasible solution":
            feasible = True
        elif terminate == "found dual feasible solution":
            feasible = False
        else:
            return self._error(
                f"SDPB inconclusive: terminateReason={terminate!r} — no "
                f"feasibility certificate exists; raise maxIterations/maxRuntime "
                f"or relax dualErrorThreshold and re-run",
                cost=time.time() - t0,
            )

        cost = time.time() - t0
        certificate = json.dumps({
            "kind": "sdpb_mixed_correlator_island",
            "dim": pt.dim, "delta_sigma": pt.delta_sigma,
            "delta_epsilon": pt.delta_epsilon,
            "orders": {"k_max": pt.k_max, "l_max": pt.l_max,
                       "m_max": pt.m_max, "n_max": pt.n_max},
            "assumptions": "single relevant Z2-odd (σ) + single relevant Z2-even (ε)",
            "in_island": feasible, "terminateReason": terminate,
        }, sort_keys=True)
        reproduces = {
            "crossing_symmetry": True, "unitarity": True, "z2_symmetry": True,
            "in_island": feasible,
        }
        # Fitness: allowed points in the island are the interesting survivors.
        fitness = 1.0 if feasible else -1.0
        verdict = VerdictGrounded(
            valid=feasible, fitness=fitness, certificate=certificate,
            reproduces_known=reproduces, cost=cost,
            details={"in_island": feasible, "terminateReason": terminate,
                     "delta_sigma": pt.delta_sigma, "delta_epsilon": pt.delta_epsilon,
                     "dim": pt.dim},
        )
        verdict.audit()
        return verdict

    # -- block tables (cached; the dominant cost) ---------------------------- #
    def _point_dirname(self, pt: IslandPoint) -> str:
        # Human-readable prefix + hash of ALL fields at full precision, so no
        # two distinct problems can ever share (and rmtree) a solve directory.
        import hashlib
        from dataclasses import asdict
        blob = json.dumps(asdict(pt), sort_keys=True, default=str)
        h = hashlib.sha1(blob.encode()).hexdigest()[:10]
        return f"pt_ds{pt.delta_sigma:.6f}_de{pt.delta_epsilon:.6f}_{h}"

    def _tables(self, pcb, pt: IslandPoint):
        base_key = (pt.dim, pt.k_max, pt.l_max, pt.m_max, pt.n_max)
        # Round the difference so float noise can't split cache entries; 10
        # decimals is far below any physically meaningful resolution here.
        d12 = round(pt.delta_epsilon - pt.delta_sigma, 10)

        entry = _MIXED_TABLE_CACHE.get(base_key)
        if entry is None:
            g1 = pcb.ConformalBlockTable(pt.dim, pt.k_max, pt.l_max,
                                         pt.m_max, pt.n_max)
            entry = {
                "f1": (pcb.ConvolvedBlockTable(g1),
                       pcb.ConvolvedBlockTable(g1, symmetric=True)),
                "d12": OrderedDict(),
            }
            _MIXED_TABLE_CACHE[base_key] = entry

        dmap = entry["d12"]
        if d12 in dmap:
            dmap.move_to_end(d12)  # LRU touch
        else:
            g2 = pcb.ConformalBlockTable(pt.dim, pt.k_max, pt.l_max, pt.m_max,
                                         pt.n_max, d12, -d12, odd_spins=True)
            g3 = pcb.ConformalBlockTable(pt.dim, pt.k_max, pt.l_max, pt.m_max,
                                         pt.n_max, -d12, -d12, odd_spins=True)
            dmap[d12] = (pcb.ConvolvedBlockTable(g2),
                         pcb.ConvolvedBlockTable(g2, symmetric=True),
                         pcb.ConvolvedBlockTable(g3))
            while len(dmap) > _D12_CACHE_MAX:
                dmap.popitem(last=False)

        f1a, f1s = entry["f1"]
        f2a, f2s, f3 = dmap[d12]
        return [f1a, f1s, f2a, f2s, f3]

    # -- the SDP (tutorial_3 transcription) --------------------------------- #
    def _solve(self, pcb, pt: IslandPoint, tab_list):
        sdp = pcb.SDP([pt.delta_sigma, pt.delta_epsilon], tab_list,
                      vector_types=_z2_info())
        # Assume: leading Z2-even scalar at Δ_ε (gap above it); a single relevant
        # Z2-odd scalar σ at Δ_σ with the next Z2-odd scalar at ≥ d.
        sdp.set_bound([0, "z2-even-l-even"], pt.delta_epsilon)
        sdp.set_bound([0, "z2-odd-l-even"], pt.dim)
        sdp.add_point([0, "z2-odd-l-even"], pt.delta_sigma)
        try:
            sdp.set_option("dualErrorThreshold", pt.dual_error_threshold)
        except Exception:  # noqa: BLE001
            pass
        sdp.iterate(name="qgse_island")  # verdict read from terminateReason below
        try:
            out = sdp.read_output(name="qgse_island")
            terminate = str(out.get("terminateReason", "unknown"))
        except Exception:  # noqa: BLE001 — never fabricate a feasibility claim
            terminate = "unknown (read_output failed)"
        _cleanup_solver_files("qgse_island")
        return terminate

    def _error(self, msg: str, cost: float = 0.0) -> VerdictGrounded:
        return VerdictGrounded(valid=False, fitness=-1e9, certificate=None,
                               reproduces_known={}, cost=cost, error=msg)
