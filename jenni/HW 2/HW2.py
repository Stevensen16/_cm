import cmath

def root2(a, b, c):
    """
    Returns the two roots of the quadratic equation:
        f(x) = a*x^2 + b*x + c
    Works for real and complex roots.
    """

    # Discriminant
    D = b*b - 4*a*c

    # Quadratic formula
    root1 = (-b + cmath.sqrt(D)) / (2*a)
    root2 = (-b - cmath.sqrt(D)) / (2*a)

    return root1, root2


# -----------------------------------------
# Test Section (Required by assignment)
# -----------------------------------------
def test(a, b, c):
    f = lambda x: a*x*x + b*x + c

    r1, r2 = root2(a, b, c)

    print("Root 1:", r1)
    print("Root 2:", r2)

    # Check f(root) is close to 0
    check1 = cmath.isclose(f(r1), 0, rel_tol=1e-9, abs_tol=0.0)
    check2 = cmath.isclose(f(r2), 0, rel_tol=1e-9, abs_tol=0.0)

    print("Is f(root1) ≈ 0 ?", check1)
    print("Is f(root2) ≈ 0 ?", check2)


# Example usage:
if __name__ == "__main__":
    # Example: x^2 + 2x + 5 = 0  (has complex roots)
    test(1, 2, 5)
