"""§6 anti-failure instrumentation for the adversarial judge funnel.

The judge is the component most likely to *quietly* destroy value, because
plausibility is anti-correlated with revolutionary truth. Guardrails are
mandatory, not optional. This module is pure bookkeeping — it never affects
scoring and never raises into the evolution loop.

It writes two artifacts into ``results_dir``:

* ``judge_audit.jsonl`` — one append-only record per verdict (for the §9
  false-reject audit and post-hoc analysis).
* ``judge_metrics.json`` — a live snapshot: accept rate, admitted-vs-rejected
  novelty/coherence spreads, fail-open count, and a mode-collapse warning that
  trips when the recent accept rate collapses (rising judge–generator agreement
  + falling diversity = the system converging on a shared delusion).
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JudgeInstrument:
    def __init__(
        self,
        results_dir: str,
        collapse_window: int = 30,
        collapse_accept_floor: float = 0.15,
    ) -> None:
        self.dir = Path(results_dir)
        self.audit_path = self.dir / "judge_audit.jsonl"
        self.metrics_path = self.dir / "judge_metrics.json"
        self.collapse_window = collapse_window
        self.collapse_accept_floor = collapse_accept_floor

        self.total = 0
        self.admitted = 0
        self.rejected = 0
        self.fail_open = 0
        self.parse_fail = 0
        self.sum_nov_admit = 0.0
        self.sum_nov_reject = 0.0
        self.sum_coh_admit = 0.0
        self.sum_coh_reject = 0.0
        self.recent = deque(maxlen=collapse_window)  # 1 admit, 0 reject
        self.collapse_warned = False

    def record(
        self,
        *,
        generation: int,
        parent_id: Optional[str],
        verdict: Any,
    ) -> None:
        """Record one verdict. Best-effort: any failure is swallowed so the
        evolution loop is never disrupted by instrumentation."""
        try:
            v = verdict.as_dict() if hasattr(verdict, "as_dict") else dict(verdict)
        except Exception:  # noqa: BLE001
            v = {}
        try:
            self._update(v)
            self._append_audit(generation, parent_id, v)
            self._write_metrics()
            self._check_collapse(generation)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"JudgeInstrument.record swallowed error: {e}")

    # -- internals ---------------------------------------------------------- #
    def _update(self, v: dict) -> None:
        self.total += 1
        admit = bool(v.get("admit"))
        nov = float(v.get("novelty", 0.0) or 0.0)
        coh = float(v.get("coherence", 0.0) or 0.0)
        if v.get("fail_open"):
            self.fail_open += 1
        if not v.get("parse_ok", True):
            self.parse_fail += 1
        if admit:
            self.admitted += 1
            self.sum_nov_admit += nov
            self.sum_coh_admit += coh
            self.recent.append(1)
        else:
            self.rejected += 1
            self.sum_nov_reject += nov
            self.sum_coh_reject += coh
            self.recent.append(0)

    def _append_audit(self, generation: int, parent_id: Optional[str], v: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "t": time.time(),
            "generation": generation,
            "parent_id": parent_id,
            **v,
        }
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _write_metrics(self) -> None:
        accept_rate = self.admitted / self.total if self.total else 0.0
        recent_rate = (sum(self.recent) / len(self.recent)) if self.recent else 0.0
        snap = {
            "total": self.total,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "accept_rate": round(accept_rate, 4),
            "recent_accept_rate": round(recent_rate, 4),
            "fail_open": self.fail_open,
            "parse_fail": self.parse_fail,
            "mean_novelty_admitted": _safe_div(self.sum_nov_admit, self.admitted),
            "mean_novelty_rejected": _safe_div(self.sum_nov_reject, self.rejected),
            "mean_coherence_admitted": _safe_div(self.sum_coh_admit, self.admitted),
            "mean_coherence_rejected": _safe_div(self.sum_coh_reject, self.rejected),
            "collapse_warning": self.collapse_warned,
            "updated_at": time.time(),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "w") as f:
            json.dump(snap, f, indent=2)

    def _check_collapse(self, generation: int) -> None:
        if len(self.recent) < self.recent.maxlen:
            return
        recent_rate = sum(self.recent) / len(self.recent)
        if recent_rate < self.collapse_accept_floor and not self.collapse_warned:
            self.collapse_warned = True
            logger.warning(
                f"[MODE-COLLAPSE WARNING] gen {generation}: judge accept rate over "
                f"last {self.recent.maxlen} proposals is {recent_rate:.2%} "
                f"(< {self.collapse_accept_floor:.0%}). The funnel may be "
                f"over-rejecting — consider loosening thresholds, injecting fresh "
                f"islands, or swapping the judge model family."
            )
        elif recent_rate >= self.collapse_accept_floor:
            self.collapse_warned = False


def _safe_div(a: float, b: int) -> float:
    return round(a / b, 4) if b else 0.0
