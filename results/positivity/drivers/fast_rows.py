"""Vectorized (numpy float64) stable high-J shaping rows.

Port of SusyR4Verifier._stable_row / _xk_stable / _rho / _G / pr1 to numpy,
vectorized over the p-grid. The jet-subtracted factored forms are
cancellation-free by construction, which is exactly what licenses float64 here;
this feeds the LP SHAPING layer only (unsound-by-construction, never trusted) --
the exact audit is untouched. `validate_fast()` gates use: it compares full rows
entrywise against the production mpmath _stable_row on jet-heavy and
oscillation-heavy samples and refuses if they disagree.

P_J via the Gegenbauer three-term recurrence, normalized P_J(1)=1 (validated
against terminating hyp2f1 to ~4e-14 across J<=120, arg in [-1.53, 1]).
"""
import numpy as np

D = 10
LAM = (D - 3) / 2.0          # Gegenbauer lambda = 7/2


def pj_rec(J, x):
    """P_J(x) = C_J^lam(x)/C_J^lam(1), three-term recurrence, vectorized."""
    x = np.asarray(x, dtype=float)
    c0 = np.ones_like(x); c1 = 2 * LAM * x
    n0 = 1.0; n1 = 2 * LAM
    if J == 0:
        return c0
    for n in range(2, J + 1):
        c0, c1 = c1, (2 * (n + LAM - 1) * x * c1 - (n + 2 * LAM - 2) * c0) / n
        n0, n1 = n1, (2 * (n + LAM - 1) * n1 - (n + 2 * LAM - 2) * n0) / n
    return c1 / n1


def pr1(r, J):
    out = 1.0
    for i in range(r):
        out *= (J * (J + 7) - i * (i + 7)) / (2.0 * (4 + i))
    return out


def rho(k, r, m2, u):
    if (k, r) == (2, 0): return (2*m2+u)/(m2*u**2*(m2+u)**2)
    if (k, r) == (2, 1): return -4/(m2*u*(u-m2)*(m2+u))
    if (k, r) == (4, 0): return (2*m2+u)/(m2**2*u**3*(m2+u)**3)
    if (k, r) == (4, 1): return 2*(-2*m2**2+3*m2*u+3*u**2)/(m2**4*u**2*(u-m2)*(m2+u)**2)
    if (k, r) == (4, 2): return -4/(m2**4*u*(u-m2)*(m2+u))
    if (k, r) == (6, 0): return (2*m2+u)/(m2**3*u**4*(m2+u)**4)
    if (k, r) == (6, 1): return -2*(2*m2**4-3*m2**3*u+2*m2**2*u**2+10*m2*u**3+5*u**4)/(m2**7*u**3*(u-m2)*(m2+u)**3)
    if (k, r) == (6, 2): return 2*(-2*m2**2+5*m2*u+5*u**2)/(m2**7*u**2*(u-m2)*(m2+u)**2)
    if (k, r) == (6, 3): return -8/(3*m2**7*u*(u-m2)*(m2+u))
    if (k, r) == (8, 0): return (2*m2+u)/(m2**4*u**5*(m2+u)**5)
    if (k, r) == (8, 1): return -2*(2*m2**6-3*m2**5*u+2*m2**4*u**2-2*m2**3*u**3-31*m2**2*u**4-36*m2*u**5-12*u**6)/(m2**10*u**4*(u-m2)*(m2+u)**4)
    if (k, r) == (8, 2): return -2*(2*m2**4-5*m2**3*u+7*m2**2*u**2+24*m2*u**3+12*u**4)/(m2**10*u**3*(u-m2)*(m2+u)**3)
    if (k, r) == (8, 3): return 4*(-2*m2**2+7*m2*u+7*u**2)/(3*m2**10*u**2*(u-m2)*(m2+u)**2)
    if (k, r) == (8, 4): return -4/(3*m2**10*u*(u-m2)*(m2+u))
    raise KeyError((k, r))


def G(k, r, m2, p2):
    if (k, r) == (2, 1): return 2*(p2 - 3*m2)/(m2**2*(m2 - p2)**2*(m2 + p2))
    if (k, r) == (4, 1): return 2*(5*m2 - 3*p2)/(m2**4*(m2 - p2)**3*(m2 + p2))
    if (k, r) == (4, 2): return -G(4, 1, m2, p2)
    if (k, r) == (6, 1): return -2*(12*m2**2 - 15*m2*p2 + 5*p2**2)/(m2**7*(m2 - p2)**4*(m2 + p2))
    if (k, r) == (6, 2): return -G(6, 1, m2, p2)
    if (k, r) == (6, 3): return -4*(7*m2**2 - 7*m2*p2 + 2*p2**2)/(3*m2**7*(m2 - p2)**4*(m2 + p2))
    if (k, r) == (8, 1): return 2*(33*m2**3 - 67*m2**2*p2 + 48*m2*p2**2 - 12*p2**3)/(m2**10*(m2 - p2)**5*(m2 + p2))
    if (k, r) == (8, 2): return -G(8, 1, m2, p2)
    if (k, r) == (8, 3): return 4*(21*m2**3 - 40*m2**2*p2 + 28*m2*p2**2 - 7*p2**3)/(3*m2**10*(m2 - p2)**5*(m2 + p2))
    if (k, r) == (8, 4): return -2*(9*m2**3 - 13*m2**2*p2 + 8*m2*p2**2 - 2*p2**3)/(3*m2**10*(m2 - p2)**5*(m2 + p2))
    raise KeyError((k, r))


def xk_fast(k, J, m2, p2, pjv):
    """_xk_stable vectorized over the p-grid (u = -p2)."""
    u = -p2
    R = k // 2
    s_ = 2.0 * p2 / m2
    jet = (J * (J + 7) * s_) < 20.0
    with np.errstate(all="ignore"):
        # --- non-jet branch: r0*Pj - sum_r pr1(r) rho(k,r) ------------------
        r0 = rho(k, 0, m2, u)
        nonjet = r0 * pjv - sum(pr1(r, J) * rho(k, r, m2, u) for r in range(R + 1))
        # --- jet branch: factored G part + r0 * tail series -----------------
        jetv = sum(pr1(r, J) * G(k, r, m2, p2) for r in range(1, R + 1))
        xm1 = 2.0 * u / m2                       # = -s_
        # tail: r0 * sum_{r>=R+1} pr1(r) xm1^r / r!, ratio-accumulated
        # bracket_t = pr1(r) xm1^r / r!  (bounded in the jet regime)
        br = pr1(R + 1, J) * xm1**(R + 1) / _fact(R + 1)
        tail = br.copy()
        for r in range(R + 1, 80):
            br = br * ((J * (J + 7) - r * (r + 7)) / (2.0 * (4 + r))) * xm1 / (r + 1)
            tail += br
            if np.max(np.abs(br)) < 1e-30 * (1.0 + np.max(np.abs(tail))):
                break
        jetv = jetv + r0 * tail
    return np.where(jet, jetv, nonjet)


def _fact(n):
    out = 1.0
    for i in range(2, n + 1):
        out *= i
    return out


def fast_row(V, J, m2, pg, pjv=None):
    """Full 39-column shaping row at (J, m2), mirroring V._stable_row."""
    if pjv is None:
        pjv = pj_rec(J, 1.0 - 2.0 * (pg * pg) / m2)
    p2 = pg * pg
    corner = np.abs(m2 - p2) < 1e-3 * m2
    safe = np.where(corner, 1.0, m2 - p2)
    safe2 = np.where(corner, 1.0, m2 * m2 - p2 * p2)
    cm2 = m2 * (2 * m2 - p2) * pjv
    c0 = np.where(corner, 0.0, (2 * m2 - p2) * pjv / safe - 2 * p2 * p2 / safe2)
    row = [np.trapezoid(pg**n * cm2, pg) for n in V.f_powers]
    row += [np.trapezoid(pg**i * c0, pg) for i in V.h_powers]
    PJp1 = J * (J + 7) / 8.0
    for k, pws in ((2, V.e_powers), (4, V.x4_powers), (6, V.x6_powers)):
        vals = np.where(corner, 0.0, xk_fast(k, J, m2, p2, pjv))
        row += [np.trapezoid(pg**i * vals, pg) for i in pws]
    if V.k_powers:
        c2i = np.where(corner, 0.0,
                       (2 * m2 - p2) * pjv / (m2 * safe**2)
                       - (p2 * p2 / m2**3) * ((4 * m2 - 3 * p2) / safe**2
                                              - 4 * p2 * PJp1 / safe2))
        row += [np.trapezoid(pg**i * c2i, pg) for i in V.k_powers]
    if V.e1_powers:
        e1v = np.where(corner, 0.0,
                       -p2 * (2 * m2 - p2) * pjv / (m2 * safe**2)
                       - m2 / (m2 + p2) * ((1 + 4 * PJp1) / safe
                                           - 2 * m2 / safe**2))
        row += [np.trapezoid(pg**i * e1v, pg) for i in V.e1_powers]
    if getattr(V, "x8_powers", ()):
        vals = np.where(corner, 0.0, xk_fast(8, J, m2, p2, pjv))
        row += [np.trapezoid(pg**i * vals, pg) for i in V.x8_powers]
    assert not getattr(V, "x10_powers", ()), "x10 not in fast path"
    if getattr(V, "e2_powers", ()):
        PJpp1 = J * (J + 7) * (J * (J + 7) - 8) / 80.0
        e2v = np.where(corner, 0.0,
                       2 * m2 * pjv * (1.0 / m2**3 + 1.0 / safe**3)
                       - m2 / (m2 + p2) * (4 * (PJp1 + 2 * PJpp1)
                                           / (m2 * safe)
                                           - 2 * (1 + 4 * PJp1) / safe**2
                                           + 4 * m2 / safe**3))
        row += [np.trapezoid(pg**i * e2v, pg) for i in V.e2_powers]
    return np.array(row, dtype=float)


def validate_fast(V, samples=((42, 1.0), (46, 30.0), (60, 9.0), (84, 45.0),
                              (120, 81.0)), tol=1e-6, log=print):
    """Gate: fast rows must agree entrywise with the production mpmath
    _stable_row (relative to each entry, with a row-scale absolute floor)."""
    import time
    pm = float(V.p_max)
    for J, m2 in samples:
        pg = np.linspace(1e-4, pm, max(240, int(2.2 * J)))
        pjv = pj_rec(J, 1.0 - 2.0 * (pg * pg) / m2)
        t0 = time.time()
        ref = np.array(V._stable_row(J, m2, pg, pjv), dtype=float)
        t1 = time.time()
        fast = fast_row(V, J, m2, pg, pjv)
        scale = np.max(np.abs(ref)) + 1e-300
        rel = np.abs(fast - ref) / np.maximum(np.maximum(np.abs(ref),
                                                         np.abs(fast)),
                                              1e-12 * scale)
        bad = rel > tol
        log("  [validate] J=%3d m2=%5.1f  max_rel=%.2e  (mpmath ref %.1fs, "
            "fast %.4fs)%s" % (J, m2, float(rel.max()), t1 - t0,
                               time.time() - t1,
                               "  FAIL" if bad.any() else ""))
        if bad.any():
            idx = int(np.argmax(rel))
            raise RuntimeError("fast row validation FAILED at (J=%d, m2=%.1f) "
                               "col %d: fast=%.6e ref=%.6e" %
                               (J, m2, idx, fast[idx], ref[idx]))
    log("  [validate] fast rows AGREE with production _stable_row -- gated OK")
    return True
