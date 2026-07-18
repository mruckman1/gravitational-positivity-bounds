"""ROUND 8 (decisive post-mortem): rebuild the round-5 stall functional
(TargetedVerifier @ 17/16, DELTA_LOWJ=3e-3, give=0.003, iter-0 then iter-1 at
c ~ 2.9870) and measure the EXACT min of E_J(w) at each refusing location.
Classification decides the endgame:
  min E < -1e-9        -> genuine violation: margins failed between grid pts;
  |min E| <= ~1e-7     -> TANGENT: physics, no audit can help; door closed;
  min E > ~1e-6        -> THIN-BUT-POSITIVE: audit-resolution limit; a sound
                          enclosure upgrade unlocks a sub-3.000 certificate.
Exact method: rational R0,R1,R2 evaluated at exact rational w (zero rounding),
transcendentals at 200 dps -- the corner-spike-verified technique."""
import sys, os, json, time
REPO = "/Users/mruckman1/Desktop/dev/quantum_gravity"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "tasks/grav_extu"))
os.chdir(REPO)
import importlib.util
import numpy as np
import sympy as sp
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_hardened_margin as thm
from tail_hardened_margin import MarginVerifier
from fast_rows import fast_row
import qgse.verifiers.gravity_susy as gs
from qgse.verifiers.gravity_positivity import _M2
from qgse.verifiers.gravity_lp import _W

DELTA_LOWJ = 3e-3          # round-5 values, exactly


def _lowj_m2s():
    return np.unique(np.concatenate([np.geomspace(1.0002, 8.0, 72),
                                     np.linspace(1.9, 4.2, 24)]))


class TargetedVerifier(MarginVerifier):        # verbatim round-5 class
    def rows_grid(self, j_max, n_xgrid, b_grid, extra_pts=()):
        rows, tags = super().rows_grid(j_max, n_xgrid, b_grid, extra_pts=())
        pm = float(self.p_max)
        add_r, add_t = [], []
        for J in range(0, 16, 2):
            pg = np.linspace(1e-4, pm, 300)
            for m2 in _lowj_m2s():
                r = fast_row(self, J, float(m2), pg)
                add_r.append(r / (np.abs(r).max() + 1e-300))
                add_t.append(("lowj", J, float(m2)))
        for (J, m2) in extra_pts:
            pg = np.linspace(1e-4, pm, max(300, int(2.2 * J)))
            r = fast_row(self, int(J), float(m2), pg)
            add_r.append(r / (np.abs(r).max() + 1e-300))
            add_t.append(("lowj", int(J), float(m2)))
        return np.vstack([rows, np.array(add_r)]), tags + add_t

    def solve(self, side, j_max=40, n_xgrid=300, b_grid=(0.25, 80.0, 240),
              extra_pts=(), cap=3000.0, give=0.10):
        rows, tags = self.rows_grid(j_max, n_xgrid, b_grid, extra_pts)
        nf, nh = len(self.f_powers), len(self.h_powers)
        ne = (len(self.e_powers) + len(self.x4_powers)
              + len(self.x6_powers) + len(self.k_powers)
              + len(self.e1_powers)
              + len(self.x8_powers) + len(self.x10_powers))
        n = nf + nh + ne
        Pf = float(self.p_max)
        gG = np.array([Pf**(p - 1) / (p - 1) for p in self.f_powers]
                      + [0.0] * (nh + ne))
        g0 = np.array([0.0] * nf
                      + [Pf**(i + 1) / (i + 1) for i in self.h_powers]
                      + [0.0] * ne)
        sgn = -1.0 if side == "upper" else 1.0
        A_eq = g0.reshape(1, -1); b_eq = np.array([sgn])
        if self.k_powers:
            off = nf + nh + len(self.e_powers) + len(self.x4_powers) \
                + len(self.x6_powers)
            g2r = np.zeros(n); g3r = np.zeros(n)
            for j, i in enumerate(self.k_powers):
                g2r[off + j] = 2.0 * Pf**(i + 1) / (i + 1)
                g3r[off + j] = Pf**(i + 3) / (i + 3)
            A_eq = np.vstack([A_eq, g2r, g3r])
            b_eq = np.append(b_eq, [0.0, 0.0])
        margins = np.zeros(len(rows))
        for i, t in enumerate(tags):
            if t[0] == "bessel":
                margins[i] = (16.0 * thm.EPS_A2 if t[1] == "asym_a2"
                              else thm.DELTA * np.abs(rows[i]).sum())
            elif t[0] == "lowj":
                margins[i] = DELTA_LOWJ * np.abs(rows[i]).sum()
        res = gs.linprog(gG, A_ub=-rows, b_ub=-margins,
                         A_eq=A_eq, b_eq=b_eq, bounds=[(-cap, cap)] * n,
                         method="highs")
        if not res.success:
            raise RuntimeError(f"LP infeasible/failed: {res.message}")
        opt = float(res.fun)
        scale = np.abs(rows).sum(axis=1) + 1e-12
        nrows = rows / scale[:, None]
        nmargins = margins / scale
        sl = np.array([[0.0] if t[0] == "bessel" else [1.0] for t in tags])
        c2 = np.zeros(n + 1); c2[-1] = -1.0
        A_ub2 = np.vstack([np.hstack([-nrows, sl]), np.append(gG, 0.0)])
        b_ub2 = np.append(-nmargins, opt + give * abs(opt) + 1e-9)
        A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
        for cap2 in (300.0, 1000.0, cap):
            r2 = gs.linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq,
                            bounds=[(-cap2, cap2)] * n + [(0, None)],
                            method="highs")
            if r2.success and r2.x[-1] > 1e-9:
                return r2.x[:n], float(gG @ r2.x[:n]), float(r2.x[-1])
        return res.x, opt, 0.0


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ev = _load(os.path.join(REPO, "tasks/grav_extu/evaluate.py"), "ev_r8")
champ = _load(os.path.join(REPO, "results/grav_extu_prod/best/main.py"), "ch_r8")
cfg = champ.run_experiment()
(fp, hp, epw, xp, x6p, kp, e1p, _pm, cap, _g, mr, clipped, ja) = ev._parse(cfg)
pm = sp.Rational(17, 16)
V = TargetedVerifier(f_powers=fp, h_powers=hp, e_powers=epw, x4_powers=xp,
                     x6_powers=x6p, k_powers=kp, e1_powers=e1p, p_max=pm)

# replicate certify's first two iterations, capturing the functional
Pq = V.p_max
nf_ = len(V.f_powers)
extra = []
arat_by_iter = []
for it in range(2):
    a, c_val, slack = V.solve("upper", j_max=40, extra_pts=extra, cap=cap,
                              give=0.003)
    print("[iter %d] LP c = %.4f slack=%.2e" % (it, c_val, slack), flush=True)
    a_rat = V._rationalize(a)
    if V.k_powers and len(V.k_powers) >= 2:
        off = (nf_ + len(V.h_powers) + len(V.e_powers)
               + len(V.x4_powers) + len(V.x6_powers))
        ks = list(V.k_powers)
        g2c = [2 * Pq**(i + 1) / (i + 1) for i in ks]
        g3c = [Pq**(i + 3) / (i + 3) for i in ks]
        m11, m12 = g2c[-2], g2c[-1]; m21, m22 = g3c[-2], g3c[-1]
        r1 = -sum(g2c[j] * a_rat[off + j] for j in range(len(ks) - 2))
        r2 = -sum(g3c[j] * a_rat[off + j] for j in range(len(ks) - 2))
        det = m11 * m22 - m12 * m21
        a_rat[off + len(ks) - 2] = (r1 * m22 - r2 * m12) / det
        a_rat[off + len(ks) - 1] = (m11 * r2 - m21 * r1) / det
    g0c = sum(q * Pq**(i + 1) / (i + 1) for q, i in
              zip(a_rat[nf_:nf_ + len(V.h_powers)], V.h_powers))
    a_rat = [-q / g0c for q in a_rat]
    cG = sum(q * Pq**(n - 1) / (n - 1)
             for q, n in zip(a_rat[:nf_], V.f_powers))
    print("  cG = %.6f" % float(cG), flush=True)
    arat_by_iter.append((float(cG), a_rat))
    fails = []
    for J in range(0, 62, 2):
        ok, why = V.audit(a_rat, J, 60)
        if not ok:
            fails.append((J, why))
            print("  audit FAIL %s" % why, flush=True)
            if len(fails) >= 3:
                break
    if not fails:
        print("  (audit PASS -- unexpected; stopping)", flush=True)
        break
    for (J, why) in fails:
        if isinstance(why, str) and "m ~" in why:
            m_star = float(why.split("m ~")[1].split(")")[0])
            extra += [(J, m_star**2), (J, (m_star * 1.02)**2),
                      (J, (m_star * 0.98)**2)]

# ---- exact min-E post-mortem on the LAST iteration's functional ------------
cG, a_rat = arat_by_iter[-1]
print("\n=== EXACT post-mortem of the c=%.4f functional ===" % cG, flush=True)
w = _W
for J, wlo, whi in ((2, 0.60, 0.80), (4, 0.45, 0.65), (6, 0.005, 0.55),
                    (0, 0.90, 1.0), (8, 0.90, 1.0)):
    cols = V._columns(J)
    integ = sum(a_rat[i] * cols[i] for i in range(len(cols)))
    Eq, r0, r1, const = V._exact_smear_integral(integ, 10, P=Pq)
    sub = {_M2: 1 / w**2}
    R0 = sp.cancel(sp.together(Eq.subs(sub) / const.subs(sub))) if Eq != 0 else sp.Integer(0)
    R1 = sp.cancel(sp.together((r0 * w).subs(sub) / const.subs(sub))) if r0 != 0 else sp.Integer(0)
    R2 = sp.cancel(sp.together(r1.subs(sub) / (2 * const.subs(sub)))) if r1 != 0 else sp.Integer(0)
    mp.mp.dps = 200
    Pfm = mp.mpf(Pq.p) / Pq.q
    best = (mp.mpf("1e40"), None)
    NPT = 1200
    for i in range(NPT + 1):
        w0 = sp.Rational(int(wlo * 10**6) + i * int((whi - wlo) * 10**6 / NPT),
                         10**6)
        if w0 <= 0 or w0 > 1:
            continue
        r0v = sp.Rational(R0.subs(w, w0)) if R0 != 0 else sp.Integer(0)
        r1v = sp.Rational(R1.subs(w, w0)) if R1 != 0 else sp.Integer(0)
        r2v = sp.Rational(R2.subs(w, w0)) if R2 != 0 else sp.Integer(0)
        x = mp.mpf(w0.p) / w0.q
        E = (mp.mpf(r0v.p) / r0v.q + (mp.mpf(r1v.p) / r1v.q) * mp.atan(Pfm * x)
             + (mp.mpf(r2v.p) / r2v.q) * mp.log(1 + (Pfm * x)**2))
        if E < best[0]:
            best = (E, w0)
    verdict = ("VIOLATION" if best[0] < mp.mpf("-1e-9") else
               "TANGENT" if best[0] < mp.mpf("1e-7") else
               "THIN-BUT-POSITIVE" if best[0] < mp.mpf("1e-3") else "HEALTHY")
    print("  J=%d on w in [%.3f,%.3f]: exact min E = %s at w=%s  -> %s"
          % (J, wlo, whi, mp.nstr(best[0], 6), best[1], verdict), flush=True)
print("done", flush=True)
