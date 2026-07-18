"""The verifier suite — the engine. Grounded, auditable checks per domain.

Build them one at a time; each is independently valuable even if the grand loop
never delivers. GR/SymPy is first (the de-risking verifier)."""

from qgse.verifiers.gr import GRVerifier, MetricSpec
from qgse.verifiers.bootstrap import BootstrapVerifier, BootstrapSpec

__all__ = ["GRVerifier", "MetricSpec", "BootstrapVerifier", "BootstrapSpec"]
