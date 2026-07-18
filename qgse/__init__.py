"""qgse — Grounded Evolutionary Search for Fundamental Physics.

The LLM proposes; grounded verifiers dispose. This package holds:

- ``interfaces``: the contracts everything depends on (§3 of the plan) —
  Candidate, VerdictGrounded, Verifier, Generator, Judge, JudgeReport.
- ``verifiers``: the engine. Grounded, auditable checks per physics domain.
  GR/SymPy is the first, de-risking verifier.
- ``judge``: the adversarial LLM funnel (a pre-filter, never an oracle) plus
  §6 anti-failure instrumentation.

Nothing in ``verifiers`` calls an LLM. The only source of truth is a
:class:`~qgse.interfaces.VerdictGrounded` carrying an auditable certificate.
"""

from qgse.interfaces import (
    Candidate,
    VerdictGrounded,
    Verifier,
    Generator,
    Judge,
    JudgeReport,
)

__all__ = [
    "Candidate",
    "VerdictGrounded",
    "Verifier",
    "Generator",
    "Judge",
    "JudgeReport",
]

__version__ = "0.1.0"
