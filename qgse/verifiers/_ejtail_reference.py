"""Stable E_J evaluator (jet-subtracted near p=0) + tail scan driver."""
import sys, json
sys.path.insert(0, "/Users/mruckman1/Desktop/dev/quantum_gravity")
import sympy as sp
import mpmath as mp

D = 10
mp.mp.dps = 40

def pr1(r, J):
    out = mp.mpf(1)
    for i in range(r):
        out *= (J * (J + 7) - i * (i + 7)) / mp.mpf(2 * (4 + i))
    return out

def pj(J, xx):
    return mp.hyp2f1(-J, J + D - 3, mp.mpf(D - 2) / 2, (1 - xx) / 2)

class EJ:
    def __init__(self, path):
        out = json.load(open(path))
        self.F, self.H = out["F"], out["H"]
        self.E, self.X4, self.X6 = out["E"], out["X4"], out["X6"]
        a = [sp.Rational(s) for s in out["a_rat"]]
        nf, nh, ne, nx4 = len(self.F), len(self.H), len(self.E), len(self.X4)
        self.af = [mp.mpf(str(float(q))) for q in a[:nf]]
        self.ah = [mp.mpf(str(float(q))) for q in a[nf:nf+nh]]
        self.ae = [mp.mpf(str(float(q))) for q in a[nf+nh:nf+nh+ne]]
        self.ax4 = [mp.mpf(str(float(q))) for q in a[nf+nh+ne:nf+nh+ne+nx4]]
        self.ax6 = [mp.mpf(str(float(q))) for q in a[nf+nh+ne+nx4:]]
        self.c = out["c"]

    def rho(self, k, r, m2, u):
        if (k, r) == (2, 0): return (2*m2+u)/(m2*u**2*(m2+u)**2)
        if (k, r) == (2, 1): return -4/(m2*u*(u-m2)*(m2+u))
        if (k, r) == (4, 0): return (2*m2+u)/(m2**2*u**3*(m2+u)**3)
        if (k, r) == (4, 1): return 2*(-2*m2**2+3*m2*u+3*u**2)/(m2**4*u**2*(u-m2)*(m2+u)**2)
        if (k, r) == (4, 2): return -4/(m2**4*u*(u-m2)*(m2+u))
        if (k, r) == (6, 0): return (2*m2+u)/(m2**3*u**4*(m2+u)**4)
        if (k, r) == (6, 1): return -2*(2*m2**4-3*m2**3*u+2*m2**2*u**2+10*m2*u**3+5*u**4)/(m2**7*u**3*(u-m2)*(m2+u)**3)
        if (k, r) == (6, 2): return 2*(-2*m2**2+5*m2*u+5*u**2)/(m2**7*u**2*(u-m2)*(m2+u)**2)
        if (k, r) == (6, 3): return -8/(3*m2**7*u*(u-m2)*(m2+u))
        raise KeyError((k, r))

    def G(self, k, r, m2, p2):
        """factored regular kernels G_{k,r} = rho0 (x-1)^r/r! - rho_r"""
        if (k, r) == (2, 1):
            return 2*(p2 - 3*m2)/(m2**2*(m2 - p2)**2*(m2 + p2))
        if (k, r) == (4, 1):
            return 2*(5*m2 - 3*p2)/(m2**4*(m2 - p2)**3*(m2 + p2))
        if (k, r) == (4, 2):
            return -self.G(4, 1, m2, p2)
        if (k, r) == (6, 1):
            return -2*(12*m2**2 - 15*m2*p2 + 5*p2**2)/(m2**7*(m2 - p2)**4*(m2 + p2))
        if (k, r) == (6, 2):
            return -self.G(6, 1, m2, p2)
        if (k, r) == (6, 3):
            return -4*(7*m2**2 - 7*m2*p2 + 2*p2**2)/(3*m2**7*(m2 - p2)**4*(m2 + p2))
        raise KeyError((k, r))

    def xk(self, k, J, m2, u, P, small):
        R = k // 2
        if not small:
            r0 = self.rho(k, 0, m2, u)
            res = sum(pr1(r, J) * self.rho(k, r, m2, u) for r in range(R + 1))
            return r0 * P - res
        # jet-subtracted, FACTORED: sum_{r=1}^{R} P^(r)(1) G_{k,r}
        #               + rho0 * sum_{r=R+1}^{RMAX} P^(r)(1)(x-1)^r/r!
        xm1 = 2 * u / m2
        r0 = self.rho(k, 0, m2, u)
        tot = mp.mpf(0)
        for r in range(1, R + 1):
            tot += pr1(r, J) * self.G(k, r, m2, -u)
        RMAX = 80
        for r in range(R + 1, RMAX + 1):
            t = pr1(r, J) * r0 * xm1**r / mp.factorial(r)
            tot += t
            if abs(t) < mp.mpf(10) ** (-60) * (1 + abs(tot)):
                break
        return tot

    def integrand(self, J, m, p):
        m2 = mp.mpf(m) ** 2
        u = -p * p
        s = -2 * u / m2
        small = (J * (J + 7)) * s < mp.mpf(20)   # jet series regime
        if small:
            RMAX = 80
            P = sum(pr1(r, J) * (2 * u / m2) ** r / mp.factorial(r)
                    for r in range(RMAX + 1))
        else:
            P = pj(J, 1 + 2 * u / m2)
        tot = mp.mpf(0)
        km2 = m2 * (2 * m2 + u) * P
        for q, n in zip(self.af, self.F):
            tot += q * p**n * km2
        k0 = (2 * m2 + u) * P / (m2 + u) - 2 * u**2 / (m2**2 - u**2)
        for q, i in zip(self.ah, self.H):
            tot += q * p**i * k0
        for q, i in zip(self.ae, self.E):
            tot += q * p**i * self.xk(2, J, m2, u, P, small)
        for q, i in zip(self.ax4, self.X4):
            tot += q * p**i * self.xk(4, J, m2, u, P, small)
        for q, i in zip(self.ax6, self.X6):
            tot += q * p**i * self.xk(6, J, m2, u, P, small)
        return tot

    def val(self, J, m):
        return mp.quad(lambda p: self.integrand(J, m, p),
                       mp.linspace(0, 1, 16))

if __name__ == "__main__":
    path = sys.argv[1]
    ej = EJ(path)
    print("c =", ej.c)
    print("sanity J<=40:")
    for J in (0, 10, 40):
        row = [float(ej.val(J, m)) for m in (1.0, 1.2, 2.0, 5.0)]
        print(f"  J={J}: " + " ".join(f"{v:+.4e}" for v in row))
    mgrid = [1.0, 1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0,
             4.0, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0, 50.0]
    print("tail scan J>40 (min over m-grid):")
    neg = []
    for J in (42, 44, 48, 56, 64, 80, 100, 128, 160, 200, 300, 400, 600, 1000):
        vals = [(float(ej.val(J, m)), m) for m in mgrid]
        mn, mmn = min(vals)
        flag = "   NEGATIVE!" if mn < 0 else ""
        print(f"  J={J:5d}: min = {mn:+.4e} at m={mmn}{flag}")
        if mn < 0:
            neg.append((J, mmn, mn))
    print("negatives:", neg)
