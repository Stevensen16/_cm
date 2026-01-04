import numpy as np
from collections import Counter

def solve_ode_general(coefficients, tol=1e-7):
    """
    Solve homogeneous linear ODE with constant coefficients:
        a_n y^(n) + a_{n-1} y^(n-1) + ... + a_1 y' + a_0 y = 0

    coefficients = [a_n, a_{n-1}, ..., a_0]

    Returns: string of general solution y(x)=...
    """

    coeffs = np.array(coefficients, dtype=float)
    if coeffs.ndim != 1 or len(coeffs) < 2:
        raise ValueError("coefficients must be a 1D list with length >= 2")

    # characteristic polynomial roots
    roots = np.roots(coeffs)

    # ---------- helpers ----------
    def near_zero(v):
        return abs(v) < tol

    def fmt(v):
        """pretty number formatting to avoid -0.0 and long floating tails"""
        if near_zero(v):
            v = 0.0
        # use ~10 significant digits, strip tiny noise
        s = f"{v:.10g}"
        # ensure something like '2' becomes '2.0' (match examples)
        if "e" not in s and "." not in s:
            s += ".0"
        return s

    def xpow(k):
        if k == 0:
            return ""
        if k == 1:
            return "x"
        return f"x^{k}"

    def exp_part(a):
        return f"e^({fmt(a)}x)"

    def cos_part(b):
        return f"cos({fmt(b)}x)"

    def sin_part(b):
        return f"sin({fmt(b)}x)"

    # ---------- group roots (merge numeric noise) ----------
    # For complex: group by (alpha, beta>0)
    # For real: group by real value
    keys = []
    for r in roots:
        if near_zero(r.imag):
            keys.append(("real", round(r.real, 10)))
        else:
            alpha = r.real
            beta = abs(r.imag)
            keys.append(("cplx", round(alpha, 10), round(beta, 10)))

    counts = Counter(keys)

    # Build a sorted list of unique keys for stable output
    # Sort by real part descending, then beta
    def sort_key(k):
        if k[0] == "real":
            return (0, -k[1], 0.0)
        else:
            return (1, -k[1], k[2])

    uniq = sorted(counts.keys(), key=sort_key)

    # ---------- construct solution ----------
    C_index = 1
    terms = []

    for k in uniq:
        if k[0] == "real":
            r = float(k[1])
            m = counts[k]
            for j in range(m):
                xp = xpow(j)
                # C_i * x^j * e^(r x)
                terms.append(f"C_{C_index}{xp}{exp_part(r)}")
                C_index += 1
        else:
            alpha = float(k[1])
            beta = float(k[2])
            m = counts[k]  # multiplicity (for each root); conjugate pair has same multiplicity
            # Since roots list contains both +iβ and -iβ, counts will be doubled.
            # Our key uses abs(beta), so it counts both together → must halve.
            m_pair = m // 2 if m >= 2 else 1

            for j in range(m_pair):
                xp = xpow(j)
                # Two constants for cos and sin
                terms.append(f"C_{C_index}{xp}{exp_part(alpha)}{cos_part(beta)}")
                C_index += 1
                terms.append(f"C_{C_index}{xp}{exp_part(alpha)}{sin_part(beta)}")
                C_index += 1

    # If polynomial is degree n, there should be n constants total.
    return "y(x) = " + " + ".join(terms)


# =========================
# 測試主程式（照你截圖的例子）
# =========================
if __name__ == "__main__":
    print("--- 實數單根範例 ---")
    coeffs1 = [1, -3, 2]   # y'' - 3y' + 2y = 0 -> roots 1,2
    print(f"方程係數: {coeffs1}")
    print(solve_ode_general(coeffs1))
    print()

    print("--- 實數重根範例 ---")
    coeffs2 = [1, -4, 4]   # (λ-2)^2
    print(f"方程係數: {coeffs2}")
    print(solve_ode_general(coeffs2))
    print()

    print("--- 複數共軛根範例 ---")
    coeffs3 = [1, 0, 4]    # y'' + 4y = 0 -> roots ±2i
    print(f"方程係數: {coeffs3}")
    print(solve_ode_general(coeffs3))
    print()

    print("--- 複數重根範例 ---")
    coeffs4 = [1, 0, 2, 0, 1]  # (λ^2+1)^2 -> roots ±i (mult 2)
    print(f"方程係數: {coeffs4}")
    print(solve_ode_general(coeffs4))
    print()

    print("--- 高階重根範例 ---")
    coeffs5 = [1, -6, 12, -8]  # (λ-2)^3
    print(f"方程係數: {coeffs5}")
    print(solve_ode_general(coeffs5))
    print()
