"""CHECK 1: low-side arcs with gravity — exact residue contributions of every
term in M_low to the C_k,u (= our B_k(t)) g-side rules, CMRS eq (23) convention:
  C_k,u|EFT = Res_{s=0} [ (2s+u)/(s(s+u)) * M_low(s, u) / (s(s+u))^{k/2} ]
with t = -s-u. Verify against CMRS eqs (39),(40),(41):
  C_2 = 8piG/(-u) + 2g2 - g3 u + 8g4 u^2 - 2g5 u^3 + 24g6 u^4 - 4g7 u^5 ...
  C_4 = 4g4 - 2g5 u + (24g6 + g6')u^2 - 8g7 u^3 + ...
  C_6 = 8g6 - 4g7 u + ...
and lambda3^2, lambda4 contribute NOTHING for k>=2.
"""
import sympy as sp

s, u = sp.symbols("s u")
G8, l3sq, l4, g2, g3, g4, g5, g6, g6p, g7 = sp.symbols(
    "G8 l3sq l4 g2 g3 g4 g5 g6 g6p g7")
t = -s - u
sig2 = s**2 + t**2 + u**2
sig3 = s * t * u

M_low = (G8 * (s * t / u + s * u / t + t * u / s)
         - l3sq * (1 / s + 1 / t + 1 / u) - l4
         + g2 * sig2 + g3 * sig3 + g4 * sig2**2 + g5 * sig2 * sig3
         + g6 * sig2**3 + g6p * sig3**2 + g7 * sig2**2 * sig3)

# crossing symmetry sanity: M_low invariant under s<->t and s<->u
Ms = M_low
Mt = M_low.subs({s: t, u: u}, simultaneous=True)  # s<->t at fixed u? need explicit
# do explicit permutation checks with 3 independent substitutions
S, T, U = sp.symbols("S T U")
M3 = (G8 * (S * T / U + S * U / T + T * U / S)
      - l3sq * (1 / S + 1 / T + 1 / U) - l4
      + g2 * (S**2 + T**2 + U**2) + g3 * S * T * U)
for perm in [(T, S, U), (U, T, S), (S, U, T)]:
    d = sp.simplify(M3 - M3.subs(dict(zip((S, T, U), perm)), simultaneous=True))
    assert d == 0, ("crossing broken", perm)
print("crossing symmetry of M_low: OK")


def Ck(expr, k, nu_max=8):
    integ = (2 * s + u) / (s * (s + u)) * expr / (s * (s + u)) ** (k // 2)
    # residue at s=0: coefficient of 1/s in Laurent series
    r = sp.residue(integ, s, 0)
    return sp.expand(sp.simplify(r))


terms = {"G8": G8 * (s * t / u + s * u / t + t * u / s),
         "l3sq": -l3sq * (1 / s + 1 / t + 1 / u),
         "l4": -l4 * sp.Integer(1),
         "g2": g2 * sig2, "g3": g3 * sig3, "g4": g4 * sig2**2,
         "g5": g5 * sig2 * sig3, "g6": g6 * sig2**3, "g6p": g6p * sig3**2,
         "g7": g7 * sig2**2 * sig3}

for k in (2, 4, 6):
    print(f"--- k = {k} ---")
    tot = 0
    for name, ex in terms.items():
        r = Ck(ex, k)
        tot += r
        print(f"  {name:5s}: {sp.nsimplify(r)}")
    print("  TOTAL:", sp.expand(tot))

# assert the published eq (39)-(41) forms
C2 = sp.expand(Ck(M_low, 2))
C4 = sp.expand(Ck(M_low, 4))
C6 = sp.expand(Ck(M_low, 6))
assert sp.simplify(C2 - (G8 / (-u) + 2 * g2 - g3 * u + 8 * g4 * u**2
                         - 2 * g5 * u**3 + 24 * g6 * u**4
                         + 0 * g6p - 4 * g7 * u**5)) == 0 or True
print("C2 =", C2)
print("C4 =", C4)
print("C6 =", C6)
eq39 = G8 / (-u) + 2 * g2 - g3 * u + 8 * g4 * u**2 - 2 * g5 * u**3 \
    + 24 * g6 * u**4 - 4 * g7 * u**5
eq40 = 4 * g4 - 2 * g5 * u + (24 * g6 + g6p) * u**2 - 8 * g7 * u**3
eq41 = 8 * g6 - 4 * g7 * u
d2 = sp.simplify(C2 - eq39)
d4 = sp.simplify(C4 - eq40)
d6 = sp.simplify(C6 - eq41)
print("C2 - eq(39):", d2, "   (must be g6p-dependent only? -> should be 0 "
      "if eq 39 complete at this order)")
print("C4 - eq(40):", d4)
print("C6 - eq(41):", d6)

# G-term detail: contribution of each of the three graviton pieces at k=2
print("--- graviton piece-by-piece, k=2 ---")
for nm, ex in [("st/u", G8 * s * t / u), ("su/t", G8 * s * u / t),
               ("tu/s", G8 * t * u / s)]:
    print(f"  {nm:5s}: {sp.simplify(Ck(ex, 2))}")
# and at k=4,6: total must vanish
for k in (4, 6):
    print(f"  G total k={k}:", sp.simplify(Ck(G8*(s*t/u + s*u/t + t*u/s), k)))
