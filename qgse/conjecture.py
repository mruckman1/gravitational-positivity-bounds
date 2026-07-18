"""Conjecture engine (Phase 4) — LLM proposes a STATEMENT, a grounded checker
disposes.

This is the move from numeric search (where the LLM ties bisection) to structural
search (where a generator is the right tool): the candidate is a mathematical
*claim* — an identity or inequality — and the verifier decides its truth status
WITHOUT any model's opinion, on three grounded levels with strict honesty about
which was reached:

  * PROVEN_TRUE  — SymPy reduces (lhs - rhs) to exactly 0 AND the proof survives
                   a high-precision numeric cross-check. An identity is asserted
                   *where both sides are defined* on the declared domain (a
                   removable singularity / pole is excluded, not part of the
                   claim); the certificate records how many points were skipped as
                   undefined so those exclusions are visible downstream.
  * REFUTED      — mpmath finds a point where the claim fails, CONFIRMED at
                   precision past the largest sub-expression magnitude: the residual
                   is BOTH relatively significant AND not shrinking toward 0 (so it
                   is a real counterexample, not catastrophic-cancellation noise);
                   or (lhs - rhs) reduces to a provably NONZERO constant; or a
                   *strict* inequality whose two sides SymPy proves equal (false
                   everywhere). A candidate that can be neither confirmed nor cleared
                   at feasible precision is INCONCLUSIVE (error), never 'supported'.
                   Disproof is rigorous.
  * SUPPORTED    — neither proven nor refuted: holds to high precision at many
                   stratified points on the declared domain but is NOT proven. A
                   CONJECTURE, never a theorem, and its support is local to the
                   sampled domain. (Same discipline as SDPB's "inconclusive": we
                   do not upgrade support to proof.)

Real-domain discipline: both sides are evaluated as complex numbers and screened
identically. If both land on the real axis they are compared in R; if both leave
it they are compared in C (so a genuine complex identity like log(x)/2=log(sqrt x)
on x<0 is still provable); only a real-vs-complex MISMATCH — the principal-branch
ambiguity behind (x**3)**(1/3)=x at x<0 — is skipped as out-of-domain, never used
to refute. Verdicts do not depend on operand order.

Fitness rewards *proven, non-trivial, bridging* statements: any NON-TRIVIAL proven
theorem outranks any merely-supported conjecture (a trivial proven restatement
scores ~0 on purpose). Non-triviality and bridging are explicitly FUNNEL
heuristics (like the section-6 judge) — they shape what is interesting, never what
is true. They are measured on the *difference* (lhs - rhs) with generically-
nonzero multiplicative factors stripped, so matched additive OR multiplicative
padding cannot inflate them (Goodhart guard).

Verified statements of the form "any X must satisfy Y" can be compiled into the
constraint store (section 5b) via :func:`to_constraint`, together with the domain
on which they were proven — the dashed feedback loop.
"""

from __future__ import annotations

import json
import math
import random
import signal
import sys
import threading
import zlib
from dataclasses import dataclass, field
from typing import Optional

import sympy as sp
import mpmath as mp

# High-precision refutation confirmation (below) evaluates at thousands of decimal
# digits and sympify may build large integer literals; both need Python's int<->str
# conversion limit raised so a large-magnitude claim is adjudicated, not crashed.
try:
    sys.set_int_max_str_digits(1_000_000)
except (AttributeError, ValueError):  # <3.11 has no limit / already permissive
    pass

_ALLOWED_FUNCS = {name: getattr(sp, name) for name in (
    "sin cos tan cot sec csc asin acos atan sinh cosh tanh "
    "exp log sqrt Abs gamma loggamma erf zeta factorial binomial"
).split()}
_ALLOWED_CONSTS = {"pi": sp.pi, "E": sp.E, "EulerGamma": sp.EulerGamma}
_RELATIONS = ("<=", ">=", "==", "<", ">", "=")  # order matters (longest first)


class _Timeout(Exception):
    pass


class _alarm:
    """Best-effort wall-clock guard for a symbolic step (main thread only)."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.ok = threading.current_thread() is threading.main_thread() and seconds > 0

    def __enter__(self):
        if self.ok:
            def handler(signum, frame):
                raise _Timeout()
            self._old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        if self.ok:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False


@dataclass
class ConjectureSpec:
    """A machine-checkable claim.

    claim:  "lhs = rhs" (identity) or with <=,>=,<,> (inequality). '=' and '=='
            both mean identity.
    variables: name -> {"domain": "real"|"positive", "lo": float, "hi": float}.
    name:   optional human label.
    """

    claim: str
    variables: dict = field(default_factory=dict)
    name: str = ""

    @staticmethod
    def from_dict(d: dict) -> "ConjectureSpec":
        return ConjectureSpec(
            claim=str(d["claim"]),
            variables=dict(d.get("variables", {})),
            name=str(d.get("name", "")),
        )


@dataclass
class ConjectureVerdict:
    status: str            # proven_true | refuted | supported | error
    fitness: float
    certificate: Optional[str]
    nontriviality: float = 0.0     # funnel heuristic in [0,1]
    bridging: float = 0.0          # funnel heuristic in [0,1]
    relation: str = "="
    error: Optional[str] = None
    details: dict = field(default_factory=dict)

    @property
    def proven(self) -> bool:
        return self.status == "proven_true"

    def to_shinka_metrics(self) -> dict:
        return {
            "combined_score": float(self.fitness),
            "public": {"status": self.status, "nontriviality": self.nontriviality,
                       "bridging": self.bridging, "relation": self.relation,
                       **{f"m_{k}": v for k, v in self.details.items()}},
            "private": {"error": self.error},
            "extra_data": {"certificate": self.certificate},
            "text_feedback": self._feedback(),
        }

    def _feedback(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        return (f"status={self.status} (fitness {self.fitness:.3g}); "
                f"nontriviality={self.nontriviality:.2f} bridging={self.bridging:.2f}; "
                + (self.details.get("counterexample_msg", "")
                   if self.status == "refuted" else
                   f"checked {self.details.get('n_points',0)} pts @ {self.details.get('dps',0)} dps"))


class ConjectureVerifier:
    def __init__(self, *, exact: bool = True, n_samples: int = 40, dps: int = 60,
                 seed: int = 20260702, rel_tol: Optional[float] = None,
                 simplify_timeout: int = 20) -> None:
        self.exact = exact
        self.n_samples = n_samples
        self.dps = dps
        self.seed = seed
        # Candidate/agreement threshold. It must sit JUST above the working-
        # precision noise floor (~10^-dps), not far above it: a residual anywhere
        # between the threshold and the noise floor is a real, resolvable difference
        # that would otherwise be mistaken for agreement (a false claim reported
        # 'supported'). We keep a ~dps/4 guard for error accumulation in complex
        # expressions; anything above it is sent to confirm(), which adjudicates
        # rigorously by escalating precision, so an over-eager threshold only costs
        # confirm() calls, never correctness.
        self.rel_tol = rel_tol if rel_tol is not None else mp.mpf(10) ** (-(dps - dps // 4))
        # tolerance for calling an mpmath value "on the real axis"; a genuine real
        # computation returns imag exactly 0, a branch excursion returns O(1) imag.
        self._imag_tol = mp.mpf(10) ** (-(dps // 4))
        self.simplify_timeout = simplify_timeout

    # -- public ------------------------------------------------------------- #
    def verify(self, spec: ConjectureSpec) -> ConjectureVerdict:
        try:
            lhs, rhs, rel, symbols, local = self._parse(spec)
        except Exception as e:  # noqa: BLE001
            return ConjectureVerdict("error", -1e9, None, error=f"parse: {e!r}")

        core = self._core(lhs, rhs)
        nontriv = self._nontriviality(core)
        bridge = self._bridging(core)
        is_identity = rel in ("=", "==")
        is_strict = rel in ("<", ">")

        # 1) exact proof. For an identity, a zero difference IS the proof. For a
        #    STRICT inequality, a zero difference means the sides are equal, so the
        #    strict inequality is provably FALSE everywhere.
        proven_equal = False
        if (is_identity or is_strict) and self.exact:
            try:
                proven_equal = self._prove_zero(lhs - rhs)
            except Exception:  # noqa: BLE001
                proven_equal = False
        proven = proven_equal and is_identity

        if is_strict and proven_equal:
            num = {"result": "refuted", "n_points": 0, "max_rel": None,
                   "msg": "sides are provably equal, so the strict inequality is "
                          "false everywhere"}
            return self._verdict(spec, rel, "refuted", proven, num, nontriv, bridge)

        # 1b) analytic refutation: if (lhs - rhs) reduces to a provably NONZERO
        #     constant, the identity is false everywhere — refute without numerics.
        #     (SymPy Add cancels a shared sub-expression, so e.g. f(x) = f(x) + 1/2
        #     becomes -1/2 even when f(x) itself is unevaluable at low precision.)
        if is_identity and not proven_equal:
            try:
                d = lhs - rhs
                if getattr(d, "is_number", False) and d.is_zero is False:
                    num = {"result": "refuted", "n_points": 0, "max_rel": None,
                           "msg": f"sides differ by the nonzero constant {d}"}
                    return self._verdict(spec, rel, "refuted", False, num,
                                         nontriv, bridge)
            except Exception:  # noqa: BLE001
                pass

        # 2) numeric refutation / support (also cross-checks a claimed proof)
        try:
            num = self._numeric(lhs, rhs, rel, symbols, spec)
        except Exception as e:  # noqa: BLE001
            num = {"result": "error", "msg": repr(e), "n_points": 0}

        res = num["result"]
        if res == "refuted":
            if proven:  # exact & numeric disagree => precision/parse bug, not a result
                return ConjectureVerdict(
                    "error", -1e9, None, nontriviality=nontriv, bridging=bridge,
                    relation=rel, error="exact-proof / numeric refutation disagree "
                    "(precision or domain bug)", details=num)
            status = "refuted"
        elif proven:
            # A proof MUST survive the numeric cross-check on at least one point. If
            # none was evaluable (undefined/mismatch across the whole domain), the
            # last line of defense never ran — do NOT ship an unchecked proof.
            if res == "supported":
                status = "proven_true"
            else:
                return ConjectureVerdict(
                    "error", -1e9, None, nontriviality=nontriv, bridging=bridge,
                    relation=rel, error="exact proof not numerically confirmable on "
                    "the declared domain (no evaluable sample point)", details=num)
        elif res == "supported":
            status = "supported"
        else:
            status = "error"

        return self._verdict(spec, rel, status, proven, num, nontriv, bridge)

    def _verdict(self, spec, rel, status, proven, num, nontriv, bridge) -> ConjectureVerdict:
        fitness = self._fitness(status, nontriv, bridge)
        cert = self._certificate(spec, rel, status, proven, num, nontriv, bridge)
        details = {"n_points": num.get("n_points", 0), "dps": self.dps,
                   "n_undefined": num.get("n_undefined", 0),
                   "max_rel_residual": num.get("max_rel"), **(
                       {"counterexample_msg": num.get("msg", "")}
                       if status == "refuted" else {})}
        return ConjectureVerdict(
            status=status, fitness=fitness, certificate=cert,
            nontriviality=nontriv, bridging=bridge, relation=rel,
            error=(num.get("msg") if status == "error" else None),
            details=details)

    def to_constraint(self, spec: ConjectureSpec, verdict: ConjectureVerdict,
                      store, kind: str = "conjectured_relation",
                      origin: str = "") -> Optional[str]:
        """Section-5b link: register a PROVEN statement as a discovered constraint
        so it can re-enter the suite. Refuses anything not exactly proven, records
        the DOMAIN on which the proof holds (a positivity-scoped identity is not a
        universal fact), and — if any sample point was undefined — flags that the
        identity holds only WHERE DEFINED (removable singularities excluded)."""
        if not verdict.proven:
            return None
        doms = {n: (v or {}).get("domain", "real") for n, v in spec.variables.items()}
        dom_str = f"  [domain: {doms}]" if doms else ""
        where = ("  [holds where both sides are defined; "
                 f"{verdict.details.get('n_undefined', 0)} sample point(s) skipped as "
                 "undefined]") if verdict.details.get("n_undefined") else ""
        cons = store.add_constraint(
            kind=kind,
            description=f"{spec.name or spec.claim}: {spec.claim}{dom_str}{where}",
            field_path="claim", op="==", value=spec.claim,
            origin=origin or "conjecture-engine (SymPy-proven)",
            confidence=1.0,
            metadata={"certificate": verdict.certificate, "claim": spec.claim,
                      "domain": doms,
                      "n_undefined": verdict.details.get("n_undefined", 0)})
        return cons.constraint_id

    def to_admissibility_constraint(
            self, spec: ConjectureSpec, verdict: ConjectureVerdict, store, *,
            field_path: str, op: Optional[str] = None, value=None,
            kind: str = "discovered_bound", origin: str = "",
            min_status: str = "supported") -> Optional[str]:
        """§5b bridge — the FUNCTIONAL constraint.

        :meth:`to_constraint` records the symbolic claim under ``field_path="claim"``
        as provenance, but that predicate does not resolve against a real verdict or
        candidate dict, so it cannot prune a search. This method instead registers a
        constraint that an :class:`~qgse.constraints.AdmissibilityGate` can actually
        run: the proven/supported relation LICENSES a structural check over a real
        candidate/verdict FIELD (e.g. a bootstrap spectrum's ``delta_phi`` or a GR
        verdict's ``reproduces_known.satisfies_efe``), and ``op``/``value`` express
        the bound. Refuses refuted/error verdicts.

        field_path: dotted path into the candidate dict or verdict view.
        op:         one of the ConstraintStore ops; defaults to the proven
                    relation ('=' -> '==', else the relation itself).
        value:      the numeric bound / target.
        min_status: 'proven_true' to require a theorem, or 'supported' to also
                    admit a high-precision-but-unproven relation (lower confidence).
        """
        rank = {"supported": 0, "proven_true": 1}
        if verdict.status not in rank or rank[verdict.status] < rank.get(min_status, 0):
            return None
        rel = verdict.relation
        op = op or ("==" if rel in ("=", "==") else rel)
        cons = store.add_constraint(
            kind=kind,
            description=f"{spec.name or spec.claim}: admissibility [{field_path} {op} {value}]",
            field_path=field_path, op=op, value=value,
            origin=origin or f"conjecture-engine ({verdict.status}): {spec.claim}",
            confidence=1.0 if verdict.proven else 0.6,
            metadata={"certificate": verdict.certificate, "claim": spec.claim,
                      "relation": rel, "status": verdict.status})
        return cons.constraint_id

    # -- parsing ------------------------------------------------------------ #
    def _parse(self, spec: ConjectureSpec):
        claim = spec.claim.strip()
        rel = next((r for r in _RELATIONS if r in claim), None)
        if rel is None:
            raise ValueError("claim has no relation (=, ==, <=, >=, <, >)")
        left, right = claim.split(rel, 1)
        # a well-formed single-relation claim leaves pure arithmetic on both
        # sides; any leftover comparison char means a chained/multi relation.
        if any(c in (left + right) for c in "<>="):
            raise ValueError("chained/multiple relations are not supported")
        local = dict(_ALLOWED_FUNCS)
        local.update(_ALLOWED_CONSTS)
        symbols = {}
        for name, spec_v in spec.variables.items():
            dom = (spec_v or {}).get("domain", "real")
            s = sp.Symbol(name, positive=True) if dom == "positive" else sp.Symbol(name, real=True)
            symbols[name] = s
            local[name] = s
        lhs = sp.sympify(left, locals=local, rational=True)
        rhs = sp.sympify(right, locals=local, rational=True)
        allowed = set(symbols.values())
        extra = {s for s in (lhs.free_symbols | rhs.free_symbols) if s not in allowed}
        if extra:
            raise ValueError(f"undeclared symbols {sorted(map(str, extra))}")
        return lhs, rhs, rel, symbols, local

    # -- exact prover ------------------------------------------------------- #
    def _prove_zero(self, expr) -> bool:
        # auto-evaluated to zero already?  (this is the syntactic-trivial case)
        if expr == 0:
            return True
        strategies = (sp.simplify, sp.expand, sp.trigsimp, sp.cancel, sp.factor,
                      sp.together, sp.radsimp,
                      lambda e: sp.simplify(sp.expand(sp.trigsimp(e))),
                      getattr(sp, "gammasimp", sp.simplify),
                      getattr(sp, "combsimp", sp.simplify),
                      getattr(sp, "logcombine", sp.simplify))
        for f in strategies:
            try:
                with _alarm(self.simplify_timeout):
                    r = f(expr)
                if r == 0 or (hasattr(r, "is_zero") and r.is_zero):
                    return True
            except (_Timeout, Exception):  # noqa: BLE001
                continue
        return False

    # -- numeric refutation / support --------------------------------------- #
    def _numeric(self, lhs, rhs, rel, symbols, spec: ConjectureSpec) -> dict:
        names = list(symbols.keys())
        syms = [symbols[n] for n in names]
        is_id = rel in ("=", "==")
        old_dps = mp.mp.dps
        try:
            mp.mp.dps = self.dps
            f_l = sp.lambdify(syms, lhs, modules="mpmath")
            f_r = sp.lambdify(syms, rhs, modules="mpmath")

            # Seed from the CANONICAL parsed form (srepr), not the raw claim text,
            # so cosmetic rephrasings (x-0.15 vs -0.15+x, +0, *1, parens) map to the
            # SAME sample set and cannot be searched for a lucky seed. Domain is
            # folded in so different boxes differ.
            key = (sp.srepr(lhs) + "|" + sp.srepr(rhs) + "|" + repr(
                [(n, (spec.variables.get(n, {}) or {}).get("domain", "real"),
                  str((spec.variables.get(n, {}) or {}).get("lo", 0.1)),
                  str((spec.variables.get(n, {}) or {}).get("hi", 2.0)))
                 for n in names]))
            seed = (int(self.seed) ^ zlib.crc32(key.encode())) & 0x7fffffff
            rng = random.Random(seed)

            # Stratified sampling per variable: split [0,1) into n_samples strata,
            # one jittered draw each, shuffled independently. Guarantees coverage,
            # so a failure region wider than one stratum is always hit.
            n = self.n_samples
            cols = {}
            for name in names:
                col = [(i + rng.random()) / n for i in range(n)]
                rng.shuffle(col)
                cols[name] = col
            draws = [[cols[name][i] for name in names] for i in range(n)]

            def point(dps, u):
                mp.mp.dps = dps
                pt = []
                for name, ui in zip(names, u):
                    v = spec.variables.get(name, {}) or {}
                    lo = mp.mpf(str(v.get("lo", 0.1)))
                    hi = mp.mpf(str(v.get("hi", 2.0)))
                    pt.append(lo + (hi - lo) * mp.mpf(str(ui)))
                return pt

            def _parts(z):
                re = z.real if hasattr(z, "real") else z
                im = z.imag if hasattr(z, "imag") else mp.mpf(0)
                return re, im

            def evaluate(dps, u):
                """('ok',a,b,regime,pt) | ('undef',...) | ('mismatch',...).

                regime is 'real' (both on the real axis) or 'complex' (both off it);
                a real-vs-complex mismatch is the branch-cut ambiguity and is skipped."""
                try:
                    pt = point(dps, u)
                    a = mp.mpmathify(f_l(*pt))
                    b = mp.mpmathify(f_r(*pt))
                except Exception:  # noqa: BLE001 — undefined here (e.g. a pole)
                    return ("undef", None, None, None, None)
                ar, ai = _parts(a)
                br, bi = _parts(b)
                if not (mp.isfinite(ar) and mp.isfinite(ai)
                        and mp.isfinite(br) and mp.isfinite(bi)):
                    return ("undef", None, None, None, None)
                a_real = abs(ai) <= self._imag_tol * (1 + abs(ar))
                b_real = abs(bi) <= self._imag_tol * (1 + abs(br))
                if a_real and b_real:
                    return ("ok", a, b, "real", pt)
                if (not a_real) and (not b_real):
                    return ("ok", a, b, "complex", pt)
                return ("mismatch", None, None, None, None)

            # Lambdify every sub-expression once, so confirm() can read the
            # magnitude of ANY sub-term (constant, coordinate, or intermediate) at a
            # point — that magnitude governs how much precision cancellation /
            # argument-reduction needs before the numeric residual can be trusted.
            sub_funcs = []
            seen = set()
            for e in (lhs, rhs):
                for node in sp.preorder_traversal(e):
                    k = id(node)
                    if k in seen:
                        continue
                    seen.add(k)
                    try:
                        sub_funcs.append(sp.lambdify(syms, node, modules="mpmath"))
                    except Exception:  # noqa: BLE001
                        pass

            def scale_mag(u):
                """Max base-10 magnitude of any sub-expression at point u (>=0)."""
                mp.mp.dps = 2 * self.dps
                try:
                    pt = point(2 * self.dps, u)
                    mx = mp.mpf(1)
                    for f in sub_funcs:
                        try:
                            v = mp.mpmathify(f(*pt))
                            av = abs(v.real if hasattr(v, "real") else v) + \
                                abs(v.imag if hasattr(v, "imag") else mp.mpf(0))
                            if mp.isfinite(av) and av > mx:
                                mx = av
                        except Exception:  # noqa: BLE001
                            pass
                    return max(0, int(mp.floor(mp.log10(mx))))
                except Exception:  # noqa: BLE001
                    return 0
                finally:
                    mp.mp.dps = self.dps

            def rel_resid(a, b):
                """Relative residual with a genuine (not floored-at-1) scale, so a
                false claim whose two sides are both tiny is not absorbed."""
                scale = max(abs(a), abs(b))
                return mp.mpf(0) if scale == 0 else abs(a - b) / scale

            def confirm(u):
                """Classify a candidate discrepancy at `u`:

                  'noise'                 — the claim holds here (cancellation noise),
                  ('genuine', a, b, reg)  — a real counterexample,
                  'inconclusive'          — cannot decide at feasible precision.

                No OPERAND magnitude is trustworthy — cancellation hides in an
                ARGUMENT (loggamma(y+1)-loggamma(y)), an INTERMEDIATE (cosh-sinh), or
                a CONSTANT phase (cos(2**4000+x)). So start precision PAST the largest
                sub-expression magnitude (scale_mag), then apply the fundamental
                test: does (a - b) CONVERGE to a stable value as precision grows, or
                keep SHRINKING toward 0? A genuine counterexample converges to a
                nonzero difference (its operands are accurate); a true identity's
                (a - b) never converges — it tracks the shrinking numerical error.
                This is denominator-free and correct even when a side is exactly 0
                (where a relative residual is ill-defined)."""
                s = scale_mag(u)
                floor = 2 * self.dps + 2 * s      # start past the sub-expression scale
                if floor > 60000:
                    return "inconclusive"         # scale beyond affordable precision
                diffs = []
                last = None
                for d in (floor, 2 * floor):
                    st, a, b, regime, _ = evaluate(d, u)
                    mp.mp.dps = self.dps
                    if st != "ok":
                        return "noise"            # not evaluable high -> don't refute
                    diffs.append(a - b)
                    last = (a, b, regime)
                d0, d1 = diffs[0], diffs[1]
                denom = max(abs(d0), abs(d1))
                if denom == 0:
                    return "noise"                # a-b exactly 0 at both -> holds
                if abs(d1 - d0) > self.rel_tol * denom:
                    return "noise"                # a-b still moving (shrinking) -> noise
                # a-b converged to a stable value; genuine only if it is a real
                # difference relative to the operand scale (not a converged ~0).
                a2, b2, reg2 = last
                if rel_resid(a2, b2) > self.rel_tol:
                    return ("genuine", a2, b2, reg2)
                return "noise"

            max_rel = mp.mpf(0)
            good = 0
            n_undef = 0
            n_mismatch = 0
            inconclusive = False
            for u in draws:
                st, a, b, regime, pt = evaluate(self.dps, u)
                if st == "undef":
                    n_undef += 1
                    continue
                if st == "mismatch" or (not is_id and regime != "real"):
                    n_mismatch += 1        # branch ambiguity / can't order in C
                    continue
                good += 1
                if is_id:
                    relr = rel_resid(a, b)
                    if relr <= self.rel_tol:
                        if relr > max_rel:
                            max_rel = relr
                        continue
                    # candidate counterexample: confirm past the sub-expression scale
                    # so a TRUE near-cancellation identity is not mis-refuted.
                    conf = confirm(u)
                    if conf == "noise":
                        continue
                    if conf == "inconclusive":
                        inconclusive = True
                        continue
                    _, ca, cb, _cr = conf
                    relr_hi = rel_resid(ca, cb)
                    return {"result": "refuted", "n_points": good,
                            "n_undefined": n_undef, "max_rel": float(relr_hi),
                            "msg": f"counterexample at "
                                   f"{dict(zip(names,[mp.nstr(x,8) for x in pt]))}: "
                                   f"lhs={mp.nstr(ca,8)} rhs={mp.nstr(cb,8)} "
                                   f"rel.resid={mp.nstr(relr_hi,3)}"}
                else:
                    ar, _ = _parts(a)
                    br, _ = _parts(b)
                    scale = max(abs(a), abs(b))
                    if scale == 0:
                        continue               # both zero -> no strict violation
                    val = ar - br
                    tol = self.rel_tol * scale
                    # numerics cannot certify strictness; treat < like <= (> like >=).
                    viol = (val > tol) if rel in ("<", "<=") else (val < -tol)
                    if not viol:
                        continue
                    conf = confirm(u)          # reject cancellation noise
                    if conf == "noise":
                        continue
                    if conf == "inconclusive":
                        inconclusive = True
                        continue
                    _, a2, b2, reg2 = conf
                    if reg2 != "real":
                        continue
                    a2r, _ = _parts(a2)
                    b2r, _ = _parts(b2)
                    sc2 = max(abs(a2), abs(b2))
                    if sc2 == 0:
                        continue
                    vh = a2r - b2r
                    th = self.rel_tol * sc2
                    viol_hi = (vh > th) if rel in ("<", "<=") else (vh < -th)
                    if viol_hi:
                        return {"result": "refuted", "n_points": good,
                                "n_undefined": n_undef, "max_rel": None,
                                "msg": f"inequality violated at "
                                       f"{dict(zip(names,[mp.nstr(x,8) for x in pt]))}: "
                                       f"lhs-rhs={mp.nstr(vh,8)}"}
            if good == 0:
                return {"result": "error", "n_points": 0, "n_undefined": n_undef,
                        "msg": "no evaluable sample point (undefined/complex/"
                               "regime-mismatch across the whole declared domain)"}
            if inconclusive:
                # a candidate counterexample could not be confirmed or cleared at
                # feasible precision -> INCONCLUSIVE, never silently 'supported'.
                return {"result": "error", "n_points": good, "n_undefined": n_undef,
                        "msg": "candidate discrepancy could not be adjudicated at "
                               "feasible precision (sub-expression scale too large)"}
            return {"result": "supported", "n_points": good, "n_undefined": n_undef,
                    "max_rel": float(max_rel)}
        finally:
            mp.mp.dps = old_dps

    # -- funnel heuristics (NOT truth) -------------------------------------- #
    def _core(self, lhs, rhs):
        """The content of the claim. Start from (lhs - rhs): SymPy's Add
        auto-cancels a syntactically-matched junk term added to both sides
        (P + (-P) -> 0). Then factor and keep only the zero-carrying factor(s),
        which strips generically-nonzero MULTIPLICATIVE padding (A*P = B*P). Only
        ever accept the stripped form when it is strictly simpler, so a clean
        identity is never mangled into something less representative."""
        try:
            diff = lhs - rhs
        except Exception:  # noqa: BLE001
            return sp.Integer(0)
        if diff == 0 or (hasattr(diff, "is_zero") and diff.is_zero):
            return diff
        try:
            with _alarm(self.simplify_timeout):
                factored = sp.factor(diff)
            if isinstance(factored, sp.Mul):
                zero = []
                for fac in factored.args:
                    if getattr(fac, "is_number", False):
                        continue          # numeric constant: not content
                    try:
                        if self._prove_zero(fac):
                            zero.append(fac)
                    except Exception:  # noqa: BLE001
                        pass
                if zero:
                    content = zero[0] if len(zero) == 1 else sp.Mul(*zero)
                    if content.count_ops() < diff.count_ops():
                        return content
        except Exception:  # noqa: BLE001
            pass
        return diff

    def _nontriviality(self, core) -> float:
        try:
            if core == 0 or (hasattr(core, "is_zero") and core.is_zero):
                return 0.0            # sides equal to SymPy -> worthless
            ops = core.count_ops()
        except Exception:  # noqa: BLE001
            return 0.0
        return float(1 - math.exp(-ops / 8.0))

    def _bridging(self, core) -> float:
        heads = set()
        try:
            for node in sp.preorder_traversal(core):
                if isinstance(node, sp.Function):
                    heads.add(type(node).__name__)
                elif node in (sp.pi, sp.E, sp.EulerGamma):
                    heads.add(str(node))
        except Exception:  # noqa: BLE001
            return 0.0
        return min(1.0, max(0, len(heads) - 1) / 3.0)

    def _fitness(self, status, nontriv, bridge) -> float:
        if status in ("refuted", "error"):
            return -1.0
        if nontriv == 0.0:            # true but trivial restatement -> ~worthless
            return 1.0
        shape = (0.3 + 0.7 * nontriv) * (0.5 + 0.5 * bridge)  # in [0.15, 1.0]
        # truth-status DOMINATES for non-trivial claims: the proven band [49,100]
        # never overlaps the supported band [<=6, 40], so no conjecture can outrank
        # a non-trivial theorem. (A trivial proven restatement deliberately scores
        # 1.0, below rich conjectures — trivial theorems are not the goal.)
        if status == "proven_true":
            return float(40.0 + 60.0 * shape)
        return float(40.0 * shape)

    def _certificate(self, spec, rel, status, proven, num, nontriv, bridge) -> Optional[str]:
        if status not in ("proven_true", "refuted"):
            return None
        doms = {n: (v or {}).get("domain", "real") for n, v in spec.variables.items()}
        n_undef = num.get("n_undefined", 0)
        proof_txt = None
        if proven:
            proof_txt = ("SymPy reduces lhs-rhs to 0 (exact); holds where both sides "
                         "are defined on the declared domain, numerically confirmed at "
                         f"{num.get('n_points', 0)} point(s)"
                         + (f" ({n_undef} skipped as undefined — removable "
                            "singularities/poles excluded)" if n_undef else ""))
        return json.dumps({
            "kind": "conjecture_verdict", "claim": spec.claim, "relation": rel,
            "status": status, "domain": doms,
            "proof": proof_txt,
            "counterexample": (num.get("msg") if status == "refuted" else None),
            "numeric": {"n_points": num.get("n_points"), "dps": self.dps,
                        "n_undefined": n_undef,
                        "max_rel_residual": num.get("max_rel")},
            "nontriviality": nontriv, "bridging": bridge,
        }, sort_keys=True)
