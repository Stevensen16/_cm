import numpy as np

# -----------------------------------------
# Numerical derivative at point x
# -----------------------------------------
def df(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)   # central difference


# -----------------------------------------
# Numerical definite integral ∫ from a to b
# Using simple Riemann sum / trapezoidal
# -----------------------------------------
def integral(f, a, b, N=10_000):
    x = np.linspace(a, b, N)
    y = f(x)
    return np.trapz(y, x)   # trapezoidal rule


# -----------------------------------------
# Test Fundamental Theorem of Calculus
# d/dx ∫₀ˣ f(t) dt  =  f(x)
# -----------------------------------------
def theorem1(f, x):
    # G(x) = ∫₀ˣ f(t) dt
    G = lambda z: integral(f, 0, z)
    
    left_side = df(G, x)    # numerical derivative
    right_side = f(x)

    print(f"Derivative of integral at x={x}: {left_side}")
    print(f"f(x) at x={x}: {right_side}")
    print(f"Difference: {abs(left_side - right_side)}")
    
    # Assert (approximately equal)
    assert np.isclose(left_side, right_side, atol=1e-4)
    print("✓ Fundamental Theorem Verified!")


# -----------------------------------------
# Example usage
# -----------------------------------------
if __name__ == "__main__":
    f = lambda t: np.sin(t)   # choose any function

    theorem1(f, x=1.0)
