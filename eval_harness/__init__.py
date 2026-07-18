"""§9 evaluation harness — prove the system rediscovers the known before we
trust it on the unknown.

Modules:
  * ``holdout``            — labeled known solutions and known non-solutions.
  * ``verifier_holdout``   — the oracle test: the verifier must accept known
                             solutions and reject fakes (necessary condition —
                             "if it cannot rediscover constraints that are
                             already known, it will not find new ones").
  * ``certificate_audit``  — re-check that every ``valid`` verdict carries an
                             auditable certificate whose residual really is ~0.
  * ``false_reject_probe`` — measure how often the adversarial judge would have
                             discarded known-true, initially-implausible cases
                             (the direct test of the plausibility-anti-correlation
                             risk, §6).
"""
