"""§3 — Key interfaces: the contracts everything else depends on.

Design invariant: a candidate is *executable or checkable*. It carries a
source artifact (code or a formal statement) so that a grounded verifier can
act on it and emit a scalar a fitness function reads, or a statement a checker
checks. This keeps grounded verification central by construction.

The one hard rule enforced here: a :class:`VerdictGrounded` is the ONLY source
of truth, and a ``valid=True`` verdict without an auditable ``certificate`` is a
bug, not a result (see :meth:`VerdictGrounded.audit`).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------- #
# Candidate
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """A proposed object: an ansatz, a Lagrangian, a set of CFT data, a metric,
    or a mathematical conjecture. Always carries the source artifact (code or a
    formal statement) so a verifier can act on it."""

    id: str
    domain: str  # "gr" | "bootstrap" | "landscape" | "conjecture" | ...
    artifact: str  # source code OR a formal statement
    lineage: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# VerdictGrounded — the only source of truth
# --------------------------------------------------------------------------- #
@dataclass
class VerdictGrounded:
    """The ONLY source of truth. Never produced by an LLM.

    Attributes
    ----------
    valid:
        Did it pass the hard check at all?
    fitness:
        Grounded scalar the evolution loop maximizes (higher is better).
        Built from residuals / feasibility margins, never from plausibility.
    certificate:
        A proof object, SDP dual, or numerical bound — auditable. A ``valid``
        verdict MUST carry one.
    reproduces_known:
        Which known constraints the candidate satisfies (GR limit, unitarity,
        S_BH, asymptotics, ...). Maps constraint-name -> bool.
    cost:
        Wall-clock seconds spent by the verifier. Verifiers are expensive;
        track it so the funnel's value can be measured.
    error:
        If the verifier could not run (parse failure, timeout), the reason.
    details:
        Domain-specific, shinka-visible sub-metrics (residual norms, margins).
    private:
        Diagnostic data hidden from the LLM loop (avoid leaking answers).
    """

    valid: bool
    fitness: float
    certificate: Optional[str]
    reproduces_known: dict = field(default_factory=dict)
    cost: float = 0.0
    error: Optional[str] = None
    details: dict = field(default_factory=dict)
    private: dict = field(default_factory=dict)

    def audit(self) -> None:
        """Enforce the certificate invariant. Raises if a valid verdict has no
        auditable artifact — that is a correctness bug (§9 certificate audit)."""
        if self.valid and not self.certificate:
            raise ValueError(
                "VerdictGrounded.valid=True with no certificate — "
                "a 'valid' with no checkable artifact is a bug, not a result."
            )

    def to_shinka_metrics(self) -> dict:
        """Serialize into exactly the dict a shinka ``aggregate_metrics_fn``
        returns, so the grounded verdict *is* the fitness signal.

        The keys match shinka's contract: ``combined_score`` (maximized),
        ``public`` (loop-visible), ``private`` (hidden), ``extra_data``
        (pickled — carries the certificate), ``text_feedback`` (str)."""
        public = {
            "valid": bool(self.valid),
            "reproduces_known": self.reproduces_known,
            "verifier_cost_s": self.cost,
            **{f"m_{k}": v for k, v in self.details.items()},
        }
        return {
            "combined_score": float(self.fitness),
            "public": public,
            "private": {"error": self.error, **self.private},
            "extra_data": {
                "certificate": self.certificate,
                "reproduces_known": self.reproduces_known,
                "details": self.details,
            },
            "text_feedback": self.feedback_text(),
        }

    def feedback_text(self) -> str:
        """Human/LLM-readable summary used as shinka ``text_feedback`` — this is
        how the generator learns *why* a candidate scored as it did."""
        if self.error:
            return f"REJECTED (verifier error): {self.error}"
        lines = [
            f"valid={self.valid}  fitness={self.fitness:.4g}  cost={self.cost:.2f}s",
        ]
        if self.reproduces_known:
            ok = [k for k, v in self.reproduces_known.items() if v]
            bad = [k for k, v in self.reproduces_known.items() if not v]
            if ok:
                lines.append("satisfies: " + ", ".join(sorted(ok)))
            if bad:
                lines.append("FAILS: " + ", ".join(sorted(bad)))
        for k, v in self.details.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Verifier — the engine
# --------------------------------------------------------------------------- #
class Verifier(ABC):
    """The engine. Each physics domain implements this against real tooling.

    ~80% of engineering effort belongs here, not in LLM plumbing. A verifier
    never consults an LLM and always returns an auditable certificate on
    ``valid=True``."""

    domain: str = "abstract"

    @abstractmethod
    def verify(self, c: Candidate) -> VerdictGrounded:
        """Run the hard check and return the grounded verdict."""

    @abstractmethod
    def known_limits(self) -> list[str]:
        """Constraints a candidate MUST satisfy to be admissible — the
        admissibility gate (e.g. '-> GR as hbar->0', '-> QFT as G->0',
        'unitary', 'anomaly-free', 'asymptotically flat'). Failing any of these
        is instant rejection, ideally *before* an expensive verifier call."""


# --------------------------------------------------------------------------- #
# Generator — LLM ensemble as mutation operators
# --------------------------------------------------------------------------- #
class Generator(ABC):
    """LLM ensemble as mutation operators. Proposes edits to parents.

    In the ShinkaEvolve fork the concrete generator IS shinka's mutation
    machinery (diff/full/cross patches over the EVOLVE-BLOCK), so this ABC is
    the seam a standalone / alternative generator would implement."""

    @abstractmethod
    def propose(self, parents: list[Candidate], context: str) -> list[Candidate]:
        ...


# --------------------------------------------------------------------------- #
# Judge — the funnel (cheap pre-filter, never authoritative)
# --------------------------------------------------------------------------- #
@dataclass
class JudgeReport:
    """A FUNNEL signal. Cheap. Never authoritative about correctness.

    Novelty is unbounded and must be capped — it only gates admission *in
    conjunction with* coherence and a checkable-claim requirement, and it never
    contributes to final fitness (which comes only from grounded verdicts)."""

    novelty: float  # embedding + LLM: distance from archive, in [0, 1]
    coherence: float  # internally consistent? in [0, 1]
    makes_checkable_claim: bool  # can a verifier even act on this?
    is_known_in_disguise: bool  # rediscovery of something already in the archive?
    admit: bool  # route to the (expensive) verifier?
    reasons: str = ""  # adversarial rationale — for false-reject auditing
    model: str = ""  # which model family judged (must differ from generator)

    def to_log_record(self, candidate_id: str) -> dict:
        rec = asdict(self)
        rec["candidate_id"] = candidate_id
        return rec


class Judge(ABC):
    """A FUNNEL, never an oracle. Only pre-filters (novelty, coherence,
    falsifiability); it never certifies correctness. Must run a *different*
    model family from the generator, in an adversarial 'find why this is wrong /
    already known' framing (§6)."""

    @abstractmethod
    def screen(self, c: Candidate) -> JudgeReport:
        ...
