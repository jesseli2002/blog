"""Minimal-atom version of the period-2 decode.

The walkthrough notebook decodes `c` from `(x1, v1, v2)` with 8 curve-vanishing
atoms found by an angle scan + least squares. That is 5 more than necessary.

This script:
  1. states the closed-form 3-atom decode and checks it on a dense grid,
  2. checks the three atoms really are one-sided (so each stays affine on all
     5 bands, which is what makes them usable),
  3. shows *why* 3 is the floor, by pushing every candidate atom into the
     3-dimensional quotient space where the counting actually happens.

Run:  python period2_decode_min.py
"""

import numpy as np

relu = lambda z: np.maximum(z, 0.0)

C_LO, C_HI = 1.0, 2.0
X_LO, X_HI = -3.0, 3.0


def v1f(x, c):
    return -2 * relu(-x - c) + 2 * relu(x - 3 + c) - c + 1.5


def v2f(x, c):
    return -4 * relu(-x - c / 2) + 4 * relu(x + c / 2 - 3) - c + 3.0


xs = np.linspace(X_LO, X_HI, 1201)
cs = np.linspace(C_LO, C_HI, 241)
X, C = np.meshgrid(xs, cs, indexing="ij")
V1, V2 = v1f(X, C), v2f(X, C)


# ----------------------------------------------------------------------------
# 1. the closed form
# ----------------------------------------------------------------------------
# Two of these are atoms the notebook's scan already found (its #1 and #7);
# only P2 is new. Written as (a, b, g, d) -> a*x1 + b*v1 + g*v2 + d:
#   P1 = (-2,  1,  0,   -1.5)   vanishes on x1 = -c/2
#   P2 = (-1,  1, -0.5,  0  )   vanishes on x1 = -c/2
#   P3 = ( 0,  1,  0,   -1.5)   vanishes on x1 = 3 - c/2
P1 = V1 - 2 * X - 1.5
P2 = V1 - 0.5 * V2 - X
P3 = V1 - 1.5

c_hat = 3 - V2 + 4 * relu(P1) - 4 * relu(P2) + 2 * relu(P3)

print("=" * 72)
print("closed form:  c = 3 - v2 + 4*relu(P1) - 4*relu(P2) + 2*relu(P3)")
print(f"  max|c_hat - c| over the grid = {np.abs(c_hat - C).max():.3e}")
print(f"  (3 ReLU neurons, vs 8 in the notebook)")


# ----------------------------------------------------------------------------
# 2. the atoms are one-sided
# ----------------------------------------------------------------------------
# Each atom must vanish on its kink curve and hold a strict, constant sign on
# each side -- that is exactly what stops relu() from opening a *new* kink
# somewhere in the interior of a band.
print("\n" + "=" * 72)
print("one-sidedness (strict sign on each side of the atom's own curve):")
for name, P, curve in [
    ("P1", P1, -C / 2),
    ("P2", P2, -C / 2),
    ("P3", P3, 3 - C / 2),
]:
    left, right = X < curve - 1e-6, X > curve + 1e-6
    lo_l, hi_l = P[left].min(), P[left].max()
    lo_r, hi_r = P[right].min(), P[right].max()
    ok = (hi_l < 0 < lo_r) or (hi_r < 0 < lo_l)
    print(f"  {name}: left in [{lo_l:+.3f}, {hi_l:+.3f}]  "
          f"right in [{lo_r:+.3f}, {hi_r:+.3f}]  ok={ok}")


# ----------------------------------------------------------------------------
# 3. why 3 is the floor
# ----------------------------------------------------------------------------
# Every signal in play is continuous, and affine on each of the 5 bands. Those
# functions form a 7-dimensional space S:
#
#     S = span{ 1, x1, c, R1, R2, R3, R4 }
#     R1 = relu(x1 + c)      R2 = relu(x1 + c/2)
#     R3 = relu(x1 + c - 3)  R4 = relu(x1 + c/2 - 3)
#
# (15 coefficients = 3 per band x 5 bands, minus 2 continuity relations per
# kink x 4 kinks = 7. Same count, but as a *space of functions*, which makes
# the realizability question askable.)
#
# What comes for free is the 4-dimensional B = span{1, x1, v1, v2}. So the
# quotient S/B is 3-dimensional, and an atom is only worth a neuron if it moves
# in that quotient. Coordinates on the quotient = the functionals killing B:
S_basis = [
    np.ones_like(X), X, C,
    relu(X + C), relu(X + C / 2), relu(X + C - 3), relu(X + C / 2 - 3),
]
S_mat = np.stack([f.ravel() for f in S_basis], axis=1)


def s_coords(F):
    """Coordinates of F in S = span{1, x, c, R1, R2, R3, R4}."""
    w, *_ = np.linalg.lstsq(S_mat, F.ravel(), rcond=None)
    resid = np.abs(S_mat @ w - F.ravel()).max()
    assert resid < 1e-8, f"not band-affine (residual {resid:.1e})"
    return w


def psi(F):
    """Project into the 3-dim quotient S / span{1, x1, v1, v2}."""
    _, _, wc, w1, w2, w3, w4 = s_coords(F)
    return np.array([wc - w3 / 2 - w4 / 4, w1 + w3, w2 + w4])


# sanity: B is killed, the target is not
for name, F in [("1", np.ones_like(X)), ("x1", X), ("v1", V1), ("v2", V2)]:
    assert np.allclose(psi(F), 0, atol=1e-7), name
target = psi(C)

print("\n" + "=" * 72)
print(f"target psi(c) = {np.round(target, 6)}   (must be hit by the atoms alone)")

# enumerate every sign-valid atom on each of the two usable curves. Half a
# circle is enough: relu(P) and relu(-P) differ by P itself, which is in B.
bases = {
    2: [(-2.0, 1.0, 0.0, -1.5), (-2.0, 0.0, 1.0, -3.0)],  # vanish on x1 = -c/2
    4: [(0.0, 1.0, 0.0, -1.5), (-2.0, 0.0, 1.0, 3.0)],    # vanish on x1 = 3-c/2
}
curve_of = {2: -C / 2, 4: 3 - C / 2}


def evaluate(co):
    a, b, g, d = co
    return a * X + b * V1 + g * V2 + d


def one_sided(j, co):
    P = evaluate(co)
    left, right = X < curve_of[j] - 1e-6, X > curve_of[j] + 1e-6
    return (P[left].max() < -1e-9 < 1e-9 < P[right].min()) or (
        P[right].max() < -1e-9 < 1e-9 < P[left].min()
    )


valid = {2: [], 4: []}
for j, (b1, b2) in bases.items():
    for ang in np.linspace(0, np.pi, 721, endpoint=False):
        co = tuple(np.cos(ang) * u + np.sin(ang) * v for u, v in zip(b1, b2))
        if one_sided(j, co):
            valid[j].append((co, psi(relu(evaluate(co)))))

print("\nquotient directions reachable by a single atom:")
for j in (2, 4):
    dirs = np.array([p for _, p in valid[j]])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    label = "x1 = -c/2" if j == 2 else "x1 = 3-c/2"
    print(f"  curve {label}: {len(valid[j])} sign-valid atoms, "
          f"they span rank {np.linalg.matrix_rank(dirs, tol=1e-7)}")
    print(f"    psi_2 component ranges over "
          f"[{dirs[:, 1].min():+.4f}, {dirs[:, 1].max():+.4f}]")
    print(f"    example unit direction: {np.round(dirs[0], 4)}")

# The punchline: curve x1=3-c/2 atoms all collapse to ONE direction, which has
# psi_2 = 0. Curve x1=-c/2 atoms span a 2-plane, but every one of them
# individually has psi_2 != 0. The target has psi_2 = 0. So:
#   - 2 atoms from x1=3-c/2  -> still rank 1, can't reach the target
#   - 1 + 1                  -> the lone psi_2 != 0 can never be cancelled
#   - 2 from x1=-c/2         -> lands inside their 2-plane, target isn't in it
# hence you need >= 3, and 2 from x1=-c/2 plus 1 from x1=3-c/2 does it.
all_atoms = valid[2] + valid[4]
best2 = np.inf
for i in range(len(all_atoms)):
    for k in range(i + 1, len(all_atoms)):
        M = np.stack([all_atoms[i][1], all_atoms[k][1]], axis=1)
        w, *_ = np.linalg.lstsq(M, target, rcond=None)
        best2 = min(best2, np.linalg.norm(M @ w - target))
print(f"\nbest 2-atom fit to the target, over all "
      f"{len(all_atoms)*(len(all_atoms)-1)//2} pairs: residual {best2:.4f}")
print("  (bounded away from 0 -> two atoms provably cannot do it)")

M3 = np.stack([psi(relu(P1)), psi(relu(P2)), psi(relu(P3))], axis=1)
w3 = np.linalg.solve(M3, target)
print(f"our 3 atoms: solve exactly with weights {np.round(w3, 6)} "
      f"(residual {np.linalg.norm(M3 @ w3 - target):.2e})")


# ----------------------------------------------------------------------------
# 4. full least-squares re-derivation, for belt and braces
# ----------------------------------------------------------------------------
feats = [np.ones_like(X), X, V1, V2, relu(P1), relu(P2), relu(P3)]
A = np.stack([f.ravel() for f in feats], axis=1)
w, *_ = np.linalg.lstsq(A, C.ravel(), rcond=None)
print("\n" + "=" * 72)
print(f"least squares on [1, x1, v1, v2, relu(P1..P3)]:")
print(f"  weights   = {np.round(w, 6)}")
print(f"  max|error| = {np.abs(A @ w - C.ravel()).max():.3e}")
print("  (expected: [3, 0, 0, -1, 4, -4, 2])")
