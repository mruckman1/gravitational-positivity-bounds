"""Mixed-correlator O(N) bootstrap — the O(N) *island* (KPSV "archipelago").

The single-correlator O(N) verifier (:mod:`qgse.verifiers.bootstrap_on`) gives one
bound per Δ_φ. The mixed system ⟨φφφφ⟩, ⟨φφss⟩, ⟨ssss⟩ — φ_i the O(N) vector, s the
leading singlet — is the fully *discriminating* object: under the assumption that φ
and s are the only relevant scalars in their channels, crossing + unitarity carve a
small closed **island** in the (Δ_φ, Δ_s) plane around the O(N) model
(Kos–Poland–Simmons-Duffin–Vichi, arXiv:1504.07997; precision islands
arXiv:1603.04436). A point outside is a genuine exclusion with an SDPB
dual-functional certificate.

Structure: the φ×φ OPE splits into singlet S (even spin), symmetric-traceless T
(even spin), antisymmetric A (odd spin); φ×s exchanges O(N) vectors V (both spin
parities); s×s exchanges singlets. Crossing gives a **7-component vectorial sum
rule**: S carries 2×2 matrices in (λ_φφO, λ_ssO), T/A/V are scalar channels — five
PyCFTBoot channels once V is split by spin parity. The five convolved block tables
are the same ones the Z2 Ising island uses (g1 with Δ12=Δ34=0; g2/g3 depending only
on d12 = Δ_s − Δ_φ), so the table cache mirrors
:mod:`qgse.verifiers.bootstrap_mixed` (per-slice reuse across a constant-d12 scan).

The crossing coefficients in :func:`_on_mixed_info` were transcribed from the two
primary papers independently, reconciled coefficient-by-coefficient, and
cross-checked by the N→1 reduction against the *validated* Z2 island structure
(`_z2_info`) and by the embedding of the *validated* single-correlator O(N) rows.
Validation of the resulting verdicts against the known 3D O(2) island lives in
``eval_harness/on_island_check.py`` — the verifier is not to be trusted until that
passes (§9 discipline).

Soundness: identical 3-way SDPB classification as every bootstrap verifier here —
primal feasible → allowed, dual feasible → excluded (certificate), anything else →
inconclusive error. Never a fake certificate.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from qgse.interfaces import Candidate, Verifier, VerdictGrounded
from qgse.verifiers.bootstrap import _ensure_env, _import_pcb, _in_dir, _REPO
from qgse.verifiers.bootstrap_mixed import _cleanup_solver_files

# Table cache, same physics as the Z2 island: g1 depends only on (dim, order);
# g2/g3 depend on externals only through d12 = Δ_s − Δ_φ. N and Δ_φ enter only the
# crossing coefficients / SDP prefactors, so one table build serves every N at a
# given (dim, order, d12).
_ONM_TABLE_CACHE: dict = {}
_D12_CACHE_MAX = 4


def _on_mixed_info(N: float):
    """The O(N) mixed-correlator crossing structure (7 equations, 5 channels).

    KPSV 1504.07997 eqs (2.8)-(2.9), reconciled against Chester 1907.05147 and
    cross-checked by (a) the N→1 reduction reproducing the validated `_z2_info`
    Ising structure quad-for-quad, and (b) rows 1-3 restricted to S(0,0)/T/A
    reproducing PyCFTBoot's validated tutorial choice-2 single-correlator O(N)
    vectors (same A-channel sign convention — no block-convention flip needed).

    Rows (component order, shared by all channels; derivative parity a,a,s,a,a,a,s):
      1: 0 = Σ_T λ² F−^{φφ,φφ} + Σ_A λ² F−^{φφ,φφ}
      2: 0 = Σ_S λ_φφS² F−^{φφ,φφ} + (1−2/N) Σ_T λ² F−^{φφ,φφ} − Σ_A λ² F−^{φφ,φφ}
      3: 0 = Σ_S λ_φφS² F+^{φφ,φφ} − (1+2/N) Σ_T λ² F+^{φφ,φφ} + Σ_A λ² F+^{φφ,φφ}
      4: 0 = Σ_S λ_ssS² F−^{ss,ss}
      5: 0 = Σ_V λ_φsV² F−^{φs,φs}
      6: 0 = Σ_S λ_φφS λ_ssS F−^{φφ,ss} + Σ_V (−1)^ℓ λ_φsV² F−^{sφ,φs}
      7: 0 = Σ_S λ_φφS λ_ssS F+^{φφ,ss} − Σ_V (−1)^ℓ λ_φsV² F+^{sφ,φs}
    N enters ONLY the two T-channel coefficients.

    Quads are [coefficient, convolved-table-index, dim-index-1, dim-index-2] over
    tab_list = [f1a, f1s, f2a, f2s, f3] (f1: g^{0,0}; f2: g^{Δs−Δφ, Δφ−Δs} for
    F∓^{sφ,φs}; f3: g^{Δφ−Δs, Δφ−Δs} for F−^{φs,φs}) and dim_list = [Δ_φ, Δ_s];
    the dim pair's average substitutes delta_ext (the "inner two" externals).
    Zero placeholders carry PARITY-CORRECT table indices (antisym rows → table 0,
    sym rows 3,7 → table 1): SDP.__init__ reads each component's derivative
    structure from the referenced table, so a wrong-parity placeholder silently
    misaligns the concatenated functional. Rebuilt fresh per call — SDP.__init__
    MUTATES vector_types in place.
    """
    # --- S channel: 2x2 in (λ_φφO, λ_ssO), even spin, FIRST (identity harvest) ---
    z = [0, 0, 0, 0]                 # zero quad, antisymmetric-parity placeholder
    zs = [0, 1, 0, 0]                # zero quad, symmetric-parity placeholder
    s1 = [[z, z], [z, z]]                                    # row 1: S absent
    s2 = [[[1, 0, 0, 0], z], [z, z]]                         # row 2: F−^{φφ,φφ} in (0,0)
    s3 = [[[1, 1, 0, 0], zs], [zs, zs]]                      # row 3: F+^{φφ,φφ} in (0,0)
    s4 = [[z, z], [z, [1, 0, 1, 1]]]                         # row 4: F−^{ss,ss} in (1,1)
    s5 = [[z, z], [z, z]]                                    # row 5: S absent
    s6 = [[z, [0.5, 0, 0, 1]], [[0.5, 0, 0, 1], z]]          # row 6: ½F−^{φφ,ss} off-diag
    s7 = [[zs, [0.5, 1, 0, 1]], [[0.5, 1, 0, 1], zs]]        # row 7: ½F+^{φφ,ss} off-diag
    vec_s = [s1, s2, s3, s4, s5, s6, s7]
    # --- T channel (traceless symmetric): 1x1, even spin; the only N-dependence ---
    vec_t = [[1, 0, 0, 0],
             [1.0 - 2.0 / N, 0, 0, 0],
             [-(1.0 + 2.0 / N), 1, 0, 0],
             list(z), list(z), list(z), list(zs)]
    # --- A channel (antisymmetric): 1x1, ODD spin (g1 must be odd_spins=True) ---
    vec_a = [[1, 0, 0, 0],
             [-1, 0, 0, 0],
             [1, 1, 0, 0],
             list(z), list(z), list(z), list(zs)]
    # --- V channel (φ×s vectors), split by spin parity for the (−1)^ℓ factor ---
    vec_ve = [list(z), list(z), list(zs), list(z),
              [1, 4, 1, 0],                                  # row 5: F−^{φs,φs}
              [1, 2, 0, 0],                                  # row 6: +F−^{sφ,φs} (ℓ even)
              [-1, 3, 0, 0]]                                 # row 7: −F+^{sφ,φs}
    vec_vo = [list(z), list(z), list(zs), list(z),
              [1, 4, 1, 0],
              [-1, 2, 0, 0],                                 # row 6: −F−^{sφ,φs} (ℓ odd)
              [1, 3, 0, 0]]                                  # row 7: +F+^{sφ,φs}
    return [[vec_s, 0, "on-S-even"],      # matrix channel first: identity source
            [vec_t, 0, "on-T-even"],
            [vec_a, 1, "on-A-odd"],
            [vec_ve, 0, "on-V-even"],
            [vec_vo, 1, "on-V-odd"]]


@dataclass
class ONIslandPoint:
    """A candidate O(N) point to test for membership in the (Δ_φ, Δ_s) island,
    under the assumption that φ and s are the only relevant scalars in the vector
    and singlet channels (T-channel: unitarity only unless t_gap is set)."""

    delta_phi: float            # O(N)-vector external scalar dimension
    delta_s: float              # leading singlet dimension
    N: float = 2.0
    dim: float = 3.0
    # assumptions:
    #   "island"       — KPSV 1504.07997 archipelago: φ and s are the only relevant
    #                    scalars in their channels, ISOLATED and TIED by
    #                    λ_φφs = λ_φsφ (the add_point `extra` OPE relation; this is
    #                    what closes the island). T channel: unitarity only.
    #   "island_no_tie"— same gaps/isolation but independent OPE coefficients
    #                    (weaker; diagnostic).
    #   "weak"         — tutorial/KPSD-1406-style: singlet continuum FROM delta_s
    #                    (s at the continuum edge, NOT isolated), vector continuum
    #                    from dim with φ isolated. The weakest island-type set;
    #                    the assumption structure under which the Z2 fractional-d
    #                    campaign validated against the literature. Use this at
    #                    fractional d (see report: isolation-type assumptions
    #                    over-exclude there — cause under investigation).
    #   "singlet_gap"  — only a continuum gap Δ ≥ delta_s in the singlet channel
    #                    (no isolation): the mixed-system analog of the
    #                    single-correlator gap bound, for consistency checks.
    #   "none"         — unitarity only (free-theory sanity: everything allowed).
    assumptions: str = "island"
    t_gap: Optional[float] = None   # optional 1603.04436-style T-scalar gap
    v_gap: Optional[float] = None   # override the vector-channel continuum bound
    #                                 (default: dim). E.g. keep 3.0 at fractional d
    #                                 to test whether a point excluded by the
    #                                 "phi' irrelevant at d" assumption reappears.
    k_max: int = 20
    l_max: int = 20
    m_max: int = 2              # mixed system: unequal externals -> even-m derivatives live
    n_max: int = 4              # research order — islands need it
    dual_error_threshold: str = "1e-15"

    @staticmethod
    def from_dict(d: dict) -> "ONIslandPoint":
        return ONIslandPoint(
            delta_phi=float(d["delta_phi"]),
            delta_s=float(d["delta_s"]),
            N=float(d.get("N", 2.0)),
            dim=float(d.get("dim", 3.0)),
            assumptions=str(d.get("assumptions", "island")),
            t_gap=(float(d["t_gap"]) if d.get("t_gap") is not None else None),
            v_gap=(float(d["v_gap"]) if d.get("v_gap") is not None else None),
            k_max=int(d.get("k_max", 20)),
            l_max=int(d.get("l_max", 20)),
            m_max=int(d.get("m_max", 2)),
            n_max=int(d.get("n_max", 4)),
            dual_error_threshold=str(d.get("dual_error_threshold", "1e-15")),
        )


class ONMixedVerifier(Verifier):
    domain = "bootstrap_on_mixed"

    def __init__(self, *, work_dir: Optional[str] = None) -> None:
        self.work_dir = Path(work_dir) if work_dir else (_REPO / "results" / "on_island_work")

    def known_limits(self) -> list[str]:
        return ["crossing_symmetry", "unitarity", "on_global_symmetry",
                "single_relevant_vector", "single_relevant_singlet"]

    def verify(self, c: Candidate) -> VerdictGrounded:
        try:
            if isinstance(c.meta.get("on_island_point"), dict):
                pt = ONIslandPoint.from_dict(c.meta["on_island_point"])
            else:
                pt = ONIslandPoint.from_dict(json.loads(c.artifact))
        except Exception as e:  # noqa: BLE001
            return self._error(f"could not parse O(N) island point: {e!r}")
        return self.verify_point(pt)

    def verify_point(self, pt: ONIslandPoint) -> VerdictGrounded:
        _ensure_env()
        pcb = _import_pcb()
        pcb.cutoff = 0  # exact poles, pinned (island verdicts need them)
        t0 = time.time()
        try:
            tabs = self._tables(pcb, pt)
            point_dir = self.work_dir / self._point_dirname(pt)
            if point_dir.exists():
                shutil.rmtree(point_dir)
            with _in_dir(point_dir):
                terminate = self._solve(pcb, pt, tabs)
        except Exception as e:  # noqa: BLE001
            return self._error(f"O(N) mixed pipeline failed: {e!r}",
                               cost=time.time() - t0)

        # 3-way classification: only these two terminations carry a certificate.
        if terminate == "found primal feasible solution":
            feasible = True
        elif terminate == "found dual feasible solution":
            feasible = False
        else:
            return self._error(
                f"SDPB inconclusive: terminateReason={terminate!r} — no feasibility "
                f"certificate; raise maxIterations/maxRuntime and re-run",
                cost=time.time() - t0)

        cost = time.time() - t0
        certificate = json.dumps({
            "kind": "sdpb_on_mixed_correlator_island",
            "global_symmetry": f"O({pt.N:g})",
            "dim": pt.dim, "N": pt.N,
            "delta_phi": pt.delta_phi, "delta_s": pt.delta_s,
            "orders": {"k_max": pt.k_max, "l_max": pt.l_max,
                       "m_max": pt.m_max, "n_max": pt.n_max},
            "assumptions": pt.assumptions, "t_gap": pt.t_gap,
            "assumptions_detail": {
                "island": "phi and s the only relevant scalars in the vector and "
                          "singlet channels, isolated and tied by "
                          "lambda_phiphis = lambda_phisphi (KPSV archipelago)",
                "island_no_tie": "phi and s the only relevant scalars, isolated, "
                                 "independent OPE coefficients",
                "singlet_gap": "continuum gap Delta >= delta_s in the singlet "
                               "channel only",
                "none": "unitarity only",
            }.get(pt.assumptions, pt.assumptions),
            "in_island": feasible, "terminateReason": terminate,
        }, sort_keys=True)
        reproduces = {"crossing_symmetry": True, "unitarity": True,
                      "on_global_symmetry": True, "in_island": feasible}
        verdict = VerdictGrounded(
            valid=feasible, fitness=(1.0 if feasible else -1.0),
            certificate=certificate, reproduces_known=reproduces, cost=cost,
            details={"in_island": feasible, "terminateReason": terminate,
                     "delta_phi": pt.delta_phi, "delta_s": pt.delta_s,
                     "N": pt.N, "dim": pt.dim})
        verdict.audit()
        return verdict

    # -- block tables (cached; identical structure to the Z2 island) --------- #
    def _point_dirname(self, pt: ONIslandPoint) -> str:
        import hashlib
        blob = json.dumps(asdict(pt), sort_keys=True, default=str)
        h = hashlib.sha1(blob.encode()).hexdigest()[:10]
        return f"onm_N{pt.N:g}_dp{pt.delta_phi:.6f}_ds{pt.delta_s:.6f}_{h}"

    def _tables(self, pcb, pt: ONIslandPoint):
        base_key = (pt.dim, pt.k_max, pt.l_max, pt.m_max, pt.n_max)
        d12 = round(pt.delta_s - pt.delta_phi, 10)

        entry = _ONM_TABLE_CACHE.get(base_key)
        if entry is None:
            # odd_spins=True on g1: the O(N) antisymmetric channel is odd-spin.
            g1 = pcb.ConformalBlockTable(pt.dim, pt.k_max, pt.l_max,
                                         pt.m_max, pt.n_max, odd_spins=True)
            entry = {
                "f1": (pcb.ConvolvedBlockTable(g1),
                       pcb.ConvolvedBlockTable(g1, symmetric=True)),
                "d12": OrderedDict(),
            }
            _ONM_TABLE_CACHE[base_key] = entry

        dmap = entry["d12"]
        if d12 in dmap:
            dmap.move_to_end(d12)
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

    # -- the SDP -------------------------------------------------------------- #
    def _solve(self, pcb, pt: ONIslandPoint, tab_list):
        sdp = pcb.SDP([pt.delta_phi, pt.delta_s], tab_list,
                      vector_types=_on_mixed_info(pt.N))
        self._set_assumptions(sdp, pt)
        try:
            sdp.set_option("dualErrorThreshold", pt.dual_error_threshold)
        except Exception:  # noqa: BLE001
            pass
        sdp.iterate(name="qgse_on_island")
        try:
            out = sdp.read_output(name="qgse_on_island")
            terminate = str(out.get("terminateReason", "unknown"))
        except Exception:  # noqa: BLE001 — never fabricate a feasibility claim
            terminate = "unknown (read_output failed)"
        _cleanup_solver_files("qgse_on_island")
        return terminate

    def _set_assumptions(self, sdp, pt: ONIslandPoint):
        """Spectrum assumptions per pt.assumptions (see ONIslandPoint).

        The "island" tie uses add_point's `extra` quintuple: fold the V-channel
        vector at Δ_φ (its (0,0) entry) into the S-channel point at Δ_s (its (0,0)
        = λ_φφs² slot), coefficient 1 — enforcing λ_φφs = λ_φsφ exactly as in
        arXiv:1603.04436 (the add_point docstring cites it). Do NOT also add the
        V point separately: the tie replaces it. All unset channels/spins keep
        PyCFTBoot's unitarity-bound defaults."""
        mode = pt.assumptions
        if mode == "none":
            pass
        elif mode == "singlet_gap":
            sdp.set_bound([0, "on-S-even"], float(pt.delta_s))
        elif mode == "weak":
            # tutorial-style: s at the edge of the singlet continuum (no point
            # block on the matrix channel); φ isolated in the vector channel.
            sdp.set_bound([0, "on-S-even"], float(pt.delta_s))
            sdp.set_bound([0, "on-V-even"],
                          float(pt.v_gap) if pt.v_gap is not None else float(pt.dim))
            sdp.add_point([0, "on-V-even"], float(pt.delta_phi))
        elif mode in ("island", "island_no_tie"):
            v_bound = float(pt.v_gap) if pt.v_gap is not None else float(pt.dim)
            sdp.set_bound([0, "on-S-even"], float(pt.dim))   # next singlet irrelevant
            sdp.set_bound([0, "on-V-even"], v_bound)         # next vector above this
            if mode == "island":
                sdp.add_point([0, "on-S-even"], float(pt.delta_s),
                              extra=[[[0, "on-V-even"], float(pt.delta_phi),
                                      [0, 0], [0, 0], 1.0]])
            else:
                sdp.add_point([0, "on-S-even"], float(pt.delta_s))
                sdp.add_point([0, "on-V-even"], float(pt.delta_phi))
        else:
            raise ValueError(f"unknown assumptions mode {mode!r}")
        if pt.t_gap is not None:
            sdp.set_bound([0, "on-T-even"], float(pt.t_gap))

    def _error(self, msg: str, cost: float = 0.0) -> VerdictGrounded:
        return VerdictGrounded(valid=False, fitness=-1e9, certificate=None,
                               reproduces_known={}, cost=cost, error=msg)
