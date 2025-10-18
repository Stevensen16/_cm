import cmath

def root3(a, b, c, d):
    """Find the roots of a cubic equation ax^3 + bx^2 + cx + d = 0."""
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero for a cubic equation.")

    # Convert to depressed cubic t^3 + pt + q = 0 using x = t - b/(3a)
    p = (3*a*c - b**2) / (3*a**2)
    q = (2*b**3 - 9*a*b*c + 27*a**2*d) / (27*a**3)

    # Compute discriminant
    Δ = (q/2)**2 + (p/3)**3

    # Cube roots of complex numbers
    def cbrt(z):
        return z**(1/3) if z.real >= 0 else -(-z)**(1/3)

    # Using cmath to handle complex results safely
    C = cmath.sqrt(Δ)
    u = (-q/2 + C)**(1/3)
    v = (-q/2 - C)**(1/3)

    # If u or v is zero, avoid division by zero
    if u == 0:
        u = 1e-15
    if v == 0:
        v = 1e-15

    # Three roots of the cubic
    omega = complex(-0.5, cmath.sqrt(3)/2)  # cube roots of unity
    t1 = u + v
    t2 = u * omega + v * omega.conjugate()
    t3 = u * omega.conjugate() + v * omega

    # Convert back to x
    x1 = t1 - b / (3*a)
    x2 = t2 - b / (3*a)
    x3 = t3 - b / (3*a)

    # Return all roots
    return x1, x2, x3


# Example usage:
a, b, c, d = 1, -6, 11, -6   # (x - 1)(x - 2)(x - 3) = 0
roots = root3(a, b, c, d)
print("Roots of the cubic equation:", roots)
