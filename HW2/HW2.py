import cmath

def root2(a, b, c):
    """Find the roots of a quadratic equation ax^2 + bx + c = 0."""
    # Calculate the discriminant
    D = b**2 - 4*a*c

    # Calculate two roots (can be complex)
    root1 = (-b + cmath.sqrt(D)) / (2 * a)
    root2 = (-b - cmath.sqrt(D)) / (2 * a)

    # Define f(x)
    f = lambda x: a*x**2 + b*x + c

    # Verify both roots using cmath.isclose (checks if f(x) ≈ 0)
    check1 = cmath.isclose(f(root1), 0, rel_tol=1e-9, abs_tol=0.0)
    check2 = cmath.isclose(f(root2), 0, rel_tol=1e-9, abs_tol=0.0)

    # Print results
    print(f"Root 1 = {root1}, f(root1) ≈ 0? {check1}")
    print(f"Root 2 = {root2}, f(root2) ≈ 0? {check2}")

    # Return the roots
    return root1, root2

# Example usage:
r1, r2 = root2(1, 2, 5)
print("Final roots:", r1, r2)
