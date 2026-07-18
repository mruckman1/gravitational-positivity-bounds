"""Constraint-discovery seam (§5b — the moonshot).

This is the only part of the system that attacks the *actual* bottleneck —
discrimination — rather than the non-bottleneck (generation). The intended
workflow (research, not yet automated) is the Bekenstein–Hawking move: force two
known theories to agree in a regime where they overlap, and read off a new
"any consistent theory must reproduce X" constraint. Each such constraint is
then *compiled into a fresh admissibility check* and fed back into the verifier
suite (the dashed loop in the architecture diagram), shrinking the admissible
space for every future candidate.

What is implemented here is the durable seam, right-sized for Phase 5:

  * :class:`ConstraintStore` — SQLite persistence for discovered constraints and
    the programs that violate them. It attaches to the *same* db file the
    evolution run uses (new tables, ``CREATE TABLE IF NOT EXISTS``, no schema
    migration), so it is backward-compatible with an existing run DB.
  * :class:`Constraint` — a structured, SAFE predicate over a
    :class:`~qgse.interfaces.VerdictGrounded` (no arbitrary code execution).
  * :func:`compile_admissibility_check` — turns a stored constraint into a
    callable gate usable *before* an expensive verifier call.

The "discovery" itself (deriving the constraint) is the research problem; this
module makes a discovered constraint a first-class, reusable object.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, List, Optional, Tuple

from qgse.interfaces import VerdictGrounded

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a is not None and a < b,
    "<=": lambda a, b: a is not None and a <= b,
    ">": lambda a, b: a is not None and a > b,
    ">=": lambda a, b: a is not None and a >= b,
    "is_true": lambda a, b: bool(a) is True,
    "is_false": lambda a, b: bool(a) is False,
}


@dataclass
class Constraint:
    """A discovered discriminating constraint: "any consistent theory must have
    ``field <op> value``". The predicate is evaluated structurally against a
    VerdictGrounded, so it is safe to persist and reload."""

    constraint_id: str
    kind: str  # e.g. "forced_consistency", "known_limit", "overlap_agreement"
    description: str
    field_path: str  # dotted path into the verdict, e.g. "reproduces_known.satisfies_efe"
    op: str  # one of _OPS
    value: Any = None
    origin: str = ""  # how it was discovered (theories forced to agree, etc.)
    confidence: float = 0.5
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)

    def check(self, verdict: VerdictGrounded) -> Tuple[bool, str]:
        """Return (satisfied, detail). A candidate that fails this is ruled out
        of the admissible space."""
        actual = _dig(_verdict_view(verdict), self.field_path)
        fn = _OPS.get(self.op)
        if fn is None:
            return True, f"unknown op {self.op!r}; treated as satisfied"
        ok = bool(fn(actual, self.value))
        return ok, f"{self.field_path}={actual!r} {self.op} {self.value!r} -> {ok}"


def compile_admissibility_check(
    constraint: Constraint,
) -> Callable[[VerdictGrounded], Tuple[bool, str]]:
    """Compile a stored constraint into a fresh admissibility check — the
    mechanism by which discovered constraints re-enter the verifier suite."""
    return constraint.check


@dataclass
class AdmissibilityReport:
    admissible: bool
    violations: List[dict] = field(default_factory=list)  # {constraint_id, description, detail}


class AdmissibilityGate:
    """The CONSUMER side of the §5b dashed loop.

    A bundle of discovered/proven constraints, each compiled into a cheap
    structural check, applied as a PRE-FILTER *before* an expensive verifier call.
    A candidate (its raw parameter dict, or a VerdictGrounded) that violates any
    constraint is inadmissible; the violation is recorded to the store so the
    admissible space provably shrinks for every future candidate — without paying
    for a solve. This is what turns a stored constraint from a passive record into
    an active gate closing the loop.
    """

    def __init__(self, store: "ConstraintStore", *, kinds: Optional[list] = None,
                 min_confidence: float = 0.0) -> None:
        self.store = store
        self._checks: List[Tuple[Constraint, Callable]] = []
        for c in store.list_constraints():
            if kinds is not None and c.kind not in kinds:
                continue
            if c.confidence < min_confidence:
                continue
            self._checks.append((c, compile_admissibility_check(c)))

    def __len__(self) -> int:
        return len(self._checks)

    def screen(self, candidate: Any) -> AdmissibilityReport:
        """Check a candidate against every compiled constraint. `candidate` may be
        a VerdictGrounded or a plain dict (e.g. a verifier spec / run_experiment
        output) — `_verdict_view` passes dicts through, so top-level parameter
        fields (delta_phi, scalar_gap, dim, ...) are checkable pre-verifier."""
        violations: List[dict] = []
        for c, chk in self._checks:
            try:
                ok, detail = chk(candidate)
            except Exception as e:  # noqa: BLE001 — a broken check must never
                ok, detail = True, f"check errored, treated as satisfied: {e!r}"
            if not ok:
                violations.append({"constraint_id": c.constraint_id,
                                   "description": c.description, "detail": detail})
        return AdmissibilityReport(admissible=(not violations), violations=violations)

    def record(self, program_id: str, report: AdmissibilityReport,
               generation: int = -1) -> None:
        for v in report.violations:
            self.store.record_violation(
                program_id, v["constraint_id"], v.get("detail", ""), generation)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
class ConstraintStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.execute("PRAGMA foreign_keys = ON")
        return c

    def _init_tables(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS discovered_constraints (
                    constraint_id TEXT PRIMARY KEY,
                    kind TEXT,
                    description TEXT,
                    field_path TEXT,
                    op TEXT,
                    value TEXT,
                    origin TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL,
                    metadata TEXT
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS program_constraint_violations (
                    program_id TEXT,
                    constraint_id TEXT,
                    violation_details TEXT,
                    generation INTEGER,
                    created_at REAL,
                    UNIQUE(program_id, constraint_id)
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_dc_kind ON discovered_constraints(kind)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_pcv_program ON "
                "program_constraint_violations(program_id)"
            )

    def add_constraint(
        self,
        *,
        kind: str,
        description: str,
        field_path: str,
        op: str,
        value: Any = None,
        origin: str = "",
        confidence: float = 0.5,
        metadata: Optional[dict] = None,
        created_at: Optional[float] = None,
    ) -> Constraint:
        cons = Constraint(
            constraint_id=str(uuid.uuid4()),
            kind=kind,
            description=description,
            field_path=field_path,
            op=op,
            value=value,
            origin=origin,
            confidence=confidence,
            created_at=created_at if created_at is not None else time.time(),
            metadata=metadata or {},
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO discovered_constraints VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    cons.constraint_id, cons.kind, cons.description, cons.field_path,
                    cons.op, json.dumps(cons.value), cons.origin, cons.confidence,
                    cons.created_at, json.dumps(cons.metadata),
                ),
            )
        return cons

    def list_constraints(self) -> List[Constraint]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT constraint_id, kind, description, field_path, op, value, "
                "origin, confidence, created_at, metadata FROM discovered_constraints"
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                Constraint(
                    constraint_id=r[0], kind=r[1], description=r[2], field_path=r[3],
                    op=r[4], value=json.loads(r[5]) if r[5] else None, origin=r[6],
                    confidence=r[7], created_at=r[8],
                    metadata=json.loads(r[9]) if r[9] else {},
                )
            )
        return out

    def record_violation(
        self, program_id: str, constraint_id: str,
        violation_details: str = "", generation: int = -1,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO program_constraint_violations "
                "VALUES (?,?,?,?,?)",
                (program_id, constraint_id, violation_details, generation, time.time()),
            )

    def constraints_affecting(self, program_id: str) -> List[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT dc.constraint_id, dc.description, pcv.violation_details "
                "FROM program_constraint_violations pcv "
                "JOIN discovered_constraints dc "
                "ON dc.constraint_id = pcv.constraint_id "
                "WHERE pcv.program_id = ?",
                (program_id,),
            ).fetchall()
        return [
            {"constraint_id": r[0], "description": r[1], "violation_details": r[2]}
            for r in rows
        ]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _verdict_view(verdict: VerdictGrounded) -> dict:
    """A read-only dict view of a verdict for structured predicates."""
    if isinstance(verdict, dict):
        return verdict
    return {
        "valid": verdict.valid,
        "fitness": verdict.fitness,
        "reproduces_known": dict(verdict.reproduces_known),
        "details": dict(verdict.details),
        "error": verdict.error,
    }


def _dig(d: Any, path: str) -> Any:
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur
