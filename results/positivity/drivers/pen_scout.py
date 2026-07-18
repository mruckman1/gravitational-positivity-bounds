"""UNITARITY SCOUT (shaping-level, UNSOUND, decision-support only):
numerical estimate of the NON-PROJECTIVE optimum -- plain positivity relaxed by
the unitarity ceiling rho_J <= rho_max=2 (NORMALIZATION PENDING U2: whether the
susy-sector density carries the eq-13 ceiling verbatim; all numbers here are
labeled with that caveat).

Penalty formula (recon-sourced, units 8piG=M=1):
  gamma_G + gamma_0 g0 >= -(rho_max/pi) Sum_{J even} n_J^{(10)}
                             Int_1^inf dm^2 m^{-8} [E_J(m^2)]_-
  n_J^{(10)} = (512 pi^4/3)(J+1)(J+2)(J+3)(J+4)(J+5)(J+6)(2J+7).
LP: min gamma_G + Sum_i W_i s_i  s.t.  E_i + s_i >= 0 (standard-grid rows,
J<=40), s_i >= 0; high-J/bessel rows stay HARD (tail sanity). W_i =
(rho_max/pi) n_J m^{-8} dm2_i (trapezoid weight on the x-grid).
Outputs: penalized optimum vs plain raw optimum, where the slack concentrates.
"""
import sys, os, json, time, math
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
import importlib.util
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_hardened_margin as thm
from tail_hardened_margin import MarginVerifier
import qgse.verifiers.gravity_susy as gs
from scipy.optimize import linprog as _lp
from qgse.verifiers.gravity_lp import _linprog_retry

RHO_MAX = 2.0


def nJ10(J):
    v = 512 * math.pi**4 / 3.0
    for i in range(1, 7):
        v *= (J + i)
    return v * (2 * J + 7)


def scout(pm, cap):
    V = MarginVerifier(f_powers=fp, h_powers=hp, e_powers=epw, x4_powers=xp,
                       x6_powers=x6p, k_powers=kp, e1_powers=e1p, p_max=pm)
    rows, tags = V.rows_grid(40, 300, (0.25, 80.0, 240))
    scale = np.abs(rows).sum(axis=1) + 1e-300
    rows_n = rows / scale[:, None]
    nf, nh = len(V.f_powers), len(V.h_powers)
    ne = (len(V.e_powers) + len(V.x4_powers) + len(V.x6_powers)
          + len(V.k_powers) + len(V.e1_powers))
    n = nf + nh + ne
    Pf = float(pm)
    gG = np.array([Pf**(p - 1) / (p - 1) for p in V.f_powers] + [0.0] * (nh + ne))
    g0 = np.array([0.0] * nf + [Pf**(i + 1) / (i + 1) for i in V.h_powers]
                  + [0.0] * ne)
    A_eq = g0.reshape(1, -1); b_eq = np.array([-1.0])
    if V.k_powers:
        off = nf + nh + len(V.e_powers) + len(V.x4_powers) + len(V.x6_powers)
        g2r = np.zeros(n); g3r = np.zeros(n)
        for j, i in enumerate(V.k_powers):
            g2r[off + j] = 2.0 * Pf**(i + 1) / (i + 1)
            g3r[off + j] = Pf**(i + 3) / (i + 3)
        A_eq = np.vstack([A_eq, g2r, g3r]); b_eq = np.append(b_eq, [0.0, 0.0])
    # margins on bessel rows (normalized units)
    margins = np.zeros(len(rows))
    for i, t in enumerate(tags):
        if t[0] == "bessel":
            margins[i] = thm.EPS_A2 if t[1] == "asym_a2" else thm.DELTA

    # --- identify standard-grid rows (J<=40 integer tags with m2) and build
    # penalty weights; group by J to get trapezoid dm2 -------------------
    byJ = {}
    for i, t in enumerate(tags):
        if isinstance(t[0], (int, np.integer)) and t[0] <= 40:
            byJ.setdefault(int(t[0]), []).append((float(t[1]), i))
    slack_idx = []; W = []
    for J, pts in sorted(byJ.items()):
        pts.sort()
        m2s = [p[0] for p in pts]
        for k, (m2, i) in enumerate(pts):
            lo = m2s[k - 1] if k > 0 else m2s[0]
            hi = m2s[k + 1] if k + 1 < len(m2s) else m2s[-1]
            dm2 = (hi - lo) / 2.0
            # E_row is the NORMALIZED row value E/scale_i; physical [E]_- =
            # scale_i * s_i, so the weight carries scale_i.
            Wi = (RHO_MAX / math.pi) * nJ10(J) * m2**(-8.0) * dm2 * scale[i]
            slack_idx.append(i); W.append(Wi)
    W = np.array(W)
    ns = len(slack_idx)
    print("  slack rows: %d (J<=40 grid); weight range [%.1e, %.1e]"
          % (ns, W.min(), W.max()), flush=True)

    # --- plain raw optimum (hard) for comparison ---------------------------
    res0 = _linprog_retry(gG, A_ub=-rows_n, b_ub=-margins, A_eq=A_eq, b_eq=b_eq,
                          bounds=[(-cap, cap)] * n, method="highs", tries=4)
    raw = float(res0.fun) if res0.success else None

    # --- penalized LP: variables [a (n), s (ns)] ---------------------------
    c = np.concatenate([gG, W])
    A_ub = -rows_n
    S = np.zeros((len(rows), ns))
    for j, i in enumerate(slack_idx):
        S[i, j] = -1.0            # -(row.a) - s_i <= 0  <=>  row.a + s_i >= 0
    A_ub = np.hstack([A_ub, S])
    b_ub = -margins
    A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], ns))])
    bounds = [(-cap, cap)] * n + [(0.0, None)] * ns
    res = _linprog_retry(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq2, b_eq=b_eq,
                         bounds=bounds, method="highs", tries=4)
    if not res.success:
        return raw, None, None
    a = res.x[:n]; s = res.x[n:]
    pen_obj = float(res.fun)                       # gamma_G + penalty
    gamma_G = float(gG @ a)
    used = [(tags[slack_idx[j]][0], tags[slack_idx[j]][1], s[j] * scale[slack_idx[j]],
             W[j] * s[j]) for j in range(ns) if s[j] > 1e-14]
    used.sort(key=lambda u: -u[3])
    return raw, (pen_obj, gamma_G, pen_obj - gamma_G), used


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_ps")
champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "ch_ps")
cfg = champ.run_experiment()
(fp, hp, epw, xp, x6p, kp, e1p, _pm, cap, _g, mr, clipped, ja) = ev._parse(cfg)

for pm in (sp.Rational(1), sp.Rational(9, 8)):
    print("\n=== SCOUT p_max=%s (rho_max=%g, NORMALIZATION PENDING U2) ==="
          % (pm, RHO_MAX), flush=True)
    t0 = time.time()
    raw, pen, used = scout(pm, cap)
    print("  plain raw optimum (hard rows):      %s"
          % ("%.6f" % raw if raw is not None else "FAIL"), flush=True)
    if pen is None:
        print("  penalized LP FAILED", flush=True)
        continue
    print("  NON-PROJECTIVE optimum (obj):        %.6f   "
          "(= gamma_G %.6f + penalty %.3e)" % pen, flush=True)
    if used:
        print("  slack concentrates at (top 8):", flush=True)
        for J, m2, sphys, cost in used[:8]:
            print("    J=%2d m2=%8.2f  [E]_-=%.3e  cost=%.3e"
                  % (J, m2, sphys, cost), flush=True)
    else:
        print("  NO slack used -> unitarity ceiling buys nothing here",
              flush=True)
    print("  (%.0fs)" % (time.time() - t0), flush=True)
print("\ndone", flush=True)
