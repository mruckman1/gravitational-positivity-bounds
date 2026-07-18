"""Adversarial judge gate — the §6 funnel, inside the evolution loop.

A cheap, decorrelated pre-filter that decides whether a candidate is worth an
expensive *grounded* verifier call. It is NEVER authoritative about correctness;
the grounded verifier is the only source of truth. Two design commitments follow
directly from the plan:

* **Decorrelation.** The judge runs a different model family from the generator
  (configured via ``adversarial_judge_models`` / ``novelty_llm_models``), in an
  adversarial "find why this is wrong / a duplicate" framing.
* **Fail-open.** A flaky judge must never *starve* the loop. On any parse/LLM
  error the gate ADMITS by default (``admit_on_parse_error=True``): the worst
  case is a wasted verifier call, whereas fail-closed could halt all progress.
  Every such event is recorded distinctly for the false-reject audit.

This module intentionally does NOT import ``qgse`` so the fork stays independent;
its verdict mirrors :class:`qgse.interfaces.JudgeReport` in spirit.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

from ..llm import AsyncLLMClient
from ..prompts.prompts_judge import ADVERSARIAL_SYSTEM_MSG, ADVERSARIAL_USER_MSG

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "novelty_min": 3.0,
    "coherence_min": 5.0,
    "must_be_checkable": True,
    "must_not_be_disguise": True,
    "admit_on_parse_error": True,  # fail OPEN — never starve the grounded verifier
}

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeVerdict:
    """A funnel signal. Cheap. Never authoritative about correctness."""

    admit: bool
    novelty: float = 0.0
    coherence: float = 0.0
    makes_checkable_claim: bool = False
    is_known_in_disguise: bool = False
    strongest_weakness: str = ""
    reasoning: str = ""
    model: str = ""
    cost: float = 0.0
    parse_ok: bool = True
    fail_open: bool = False  # admitted because the judge itself errored
    thresholds: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdversarialJudgeGate:
    def __init__(
        self,
        async_llm_client: AsyncLLMClient,
        thresholds: Optional[Dict[str, Any]] = None,
        language: str = "python",
        max_code_chars: int = 12000,
    ) -> None:
        self.client = async_llm_client
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.language = language
        self.max_code_chars = max_code_chars

    # -- main entry --------------------------------------------------------- #
    async def assess(
        self,
        *,
        exec_fname: str,
        code_diff: Optional[str] = None,
        parent_program: Optional[Any] = None,
    ) -> JudgeVerdict:
        candidate_code = self._read(exec_fname)
        if not candidate_code:
            # Nothing to judge -> admit (verifier will reject an empty candidate).
            return JudgeVerdict(
                admit=True, parse_ok=False, fail_open=True,
                strongest_weakness="could not read candidate code",
                thresholds=self.thresholds,
            )

        parent_code = getattr(parent_program, "code", "") or ""
        user_msg = ADVERSARIAL_USER_MSG.format(
            language=self.language,
            parent_code=self._clip(parent_code),
            candidate_code=self._clip(candidate_code),
            code_diff=self._clip(code_diff or "(none)", 4000),
        )

        content, cost = await self._query(user_msg)
        if content is None:
            return self._fail_open(cost, "judge LLM returned no content")

        parsed = self._parse(content)
        if parsed is None:
            return self._fail_open(cost, "judge response was not valid JSON", raw=content)

        return self._apply_thresholds(parsed, cost)

    # -- internals ---------------------------------------------------------- #
    async def _query(self, user_msg: str):
        try:
            response = await self.client.query(
                msg=user_msg, system_msg=ADVERSARIAL_SYSTEM_MSG
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Adversarial judge query failed: {e}")
            return None, 0.0
        if response is None:
            return None, 0.0
        content = getattr(response, "content", None)
        cost = getattr(response, "cost", 0.0) or 0.0
        model = getattr(response, "model_name", "") or getattr(response, "model", "")
        self._last_model = model
        return content, cost

    def _apply_thresholds(self, p: Dict[str, Any], cost: float) -> JudgeVerdict:
        t = self.thresholds
        novelty = _as_float(p.get("novelty"))
        coherence = _as_float(p.get("coherence"))
        checkable = bool(p.get("makes_checkable_claim", False))
        disguise = bool(p.get("is_known_in_disguise", False))

        admit = (
            novelty >= t["novelty_min"]
            and coherence >= t["coherence_min"]
            and (checkable or not t["must_be_checkable"])
            and (not disguise or not t["must_not_be_disguise"])
        )
        return JudgeVerdict(
            admit=admit,
            novelty=novelty,
            coherence=coherence,
            makes_checkable_claim=checkable,
            is_known_in_disguise=disguise,
            strongest_weakness=str(p.get("strongest_weakness", ""))[:500],
            reasoning=str(p.get("reasoning", ""))[:1000],
            model=getattr(self, "_last_model", ""),
            cost=cost,
            parse_ok=True,
            thresholds=dict(self.thresholds),
        )

    def _fail_open(self, cost: float, reason: str, raw: str = "") -> JudgeVerdict:
        admit = bool(self.thresholds.get("admit_on_parse_error", True))
        logger.warning(f"Judge gate fail-open ({reason}); admit={admit}")
        return JudgeVerdict(
            admit=admit,
            strongest_weakness=reason,
            reasoning=(raw or "")[:500],
            model=getattr(self, "_last_model", ""),
            cost=cost,
            parse_ok=False,
            fail_open=True,
            thresholds=dict(self.thresholds),
        )

    @staticmethod
    def _parse(content: str) -> Optional[Dict[str, Any]]:
        content = content.strip()
        # strip code fences if present
        if content.startswith("```"):
            content = content.strip("`")
            content = content.split("\n", 1)[-1] if "\n" in content else content
        try:
            return json.loads(content)
        except Exception:  # noqa: BLE001
            m = _JSON_RE.search(content)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return None

    def _clip(self, s: str, limit: Optional[int] = None) -> str:
        limit = limit or self.max_code_chars
        if s is None:
            return ""
        return s if len(s) <= limit else s[:limit] + "\n... [truncated]"

    @staticmethod
    def _read(exec_fname: str) -> Optional[str]:
        try:
            with open(exec_fname, "r") as f:
                return f.read()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Judge gate could not read {exec_fname}: {e}")
            return None


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
