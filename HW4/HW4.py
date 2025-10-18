import cmath
import random
from typing import List, Tuple
import numpy as np

# ---------- Utilities ----------
def poly_eval(c: List[complex], x: complex) -> complex:
    """Evaluate p(x) = sum_{i=0}^n c[i] x^i using Horner."""
    acc = 0j
    for a in reversed(c):
        acc = acc * x + a
    return acc

def normalize(c: List[complex]) -> List[complex]:
    """Make leading coefficient = 1 (monic)."""
    if not c or all(a == 0 for a in c):
        raise ValueError("Polynomial is zero.")
    # Remove trailing zeros at the top degree if any
    i = len(c) - 1
    while i > 0 and c[i] == 0:
        i -= 1
    c = c[: i + 1]
    lead = c[-1]
    return [a / lead for a in c]

# ---------- Durand–Kerner (Weierstrass) ----------
def durand_kerner(c: List[complex], tol: float = 1e-12, max_iter: int = 2000) -> List[complex]:
    """
    Find all roots of polynomial with coefficients c[i] for x^i (c[0] const).
    Returns a list of complex roots (no guaranteed order).
    """
    c = normalize(c)
    n = len(c) - 1
    if n == 0:
        return []  # constant polynomial
    if n == 1:
        # ax + b = 0 with monic a=1 -> x = -c[0]
        return [-c[0]]

    # Initial guesses: distinct points on a circle
    radius = 1.0 + max(1.0, max(abs(a) for a in c[:-1]))
    roots = [radius * cmath.exp(2j * cmath.pi * k / n) for k in range(n)]

    for _ in range(max_iter):
        prev = roots.copy()
        converged = True
        for k in range(n):
            rk = roots[k]
            denom = 1+0j
            for j in range(n):
                if j == k:
                    continue
                diff = rk - roots[j]
                # prevent division by a near-zero difference
                if abs(diff) < 1e-18:
                    diff = complex(diff.real + 1e-16 * (random.random()-0.5),
                                   diff.imag + 1e-16 * (random.random()-0.5))
                denom *= diff
            fk = poly_eval(c, rk)
            rk_new = rk - fk / denom
            roots[k] = rk_new
            if abs(rk_new - rk) > tol:
                converged = False
        if converged:
            break

    return roots

# ---------- Companion-matrix method ----------
def companion_roots(c: List[complex]) -> List[complex]:
    """Roots via eigenvalues of the companion matrix (requires NumPy)."""
    c = normalize(c)
    n = len(c) - 1
    if n <= 0:
        return []
    # Companion matrix of monic polynomial x^n + a_{n-1} x^{n-1} + ... + a_0
    a = [-c[i] for i in range(n)]  # a_0..a_{n-1}
    M = np.zeros((n, n), dtype=complex)
    M[1:, :-1] = np.eye(n - 1, dtype=complex)
    M[:, -1] = a
    vals = np.linalg.eigvals(M)
    return list(vals)

# ---------- Public API ----------
def roots(c: List[complex], method: str = "durand-kerner", verify: bool = True
         ) -> Tuple[List[complex], List[float]]:
    """
    Compute all roots of polynomial with coefficients c[i] for x^i.
    method: 'durand-kerner' (pure Python) or 'companion' (NumPy).
    Returns (roots, residuals) where residuals are |p(root)|.
    """
    if method == "durand-kerner":
        rs = durand_kerner(c)
    elif method == "companion":
        rs = companion_roots(c)
    else:
        raise ValueError("Unknown method.")

    residuals = [abs(poly_eval(c, r)) for r in rs] if verify else []
    return rs, residuals

# ---------- Examples ----------
if __name__ == "__main__":
    # 1) Degree 5 (no closed form): x^5 - 1
    c1 = [-1, 0, 0, 0, 0, 1]  # -1 + 0x + ... + 1*x^5
    r1, e1 = roots(c1, method="durand-kerner")
    print("Roots for x^5 - 1:")
    for r, e in zip(r1, e1):
        print(f"  {r:.12g}   |p(r)|={e:.2e}")

    # 2) Random degree 7 polynomial
    c2 = [3, -2, 5, 0, -1, 4, 0, 1]  # 3 -2x + 5x^2 - x^4 + 4x^5 + x^7
    r2, e2 = roots(c2, method="companion")
    print("\nDegree-7 (companion) roots residuals:")
    print([f"{ei:.2e}" for ei in e2])
