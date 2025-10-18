import numpy as np
from scipy.integrate import quad

# Derivative approximation
def df(f, x, h=1e-5):
    """Numerically compute derivative of f at x."""
    return (f(x + h) - f(x - h)) / (2 * h)

# Integral from a to b
def integral(f, a, b):
    """Compute definite integral of f from a to b."""
    result, _ = quad(f, a, b)
    return result

# Fundamental theorem of calculus check
def theorem1(f, x):
    """Verify that d/dx ∫₀ˣ f(t) dt = f(x)."""
    F = lambda x_val: integral(f, 0, x_val)
    derivative_at_x = df(F, x)
    print(f"Derivative at x={x}: {derivative_at_x:.6f}, f(x): {f(x):.6f}")
    return np.isclose(derivative_at_x, f(x), atol=1e-4)

# Example function: f(t) = t^2
f = lambda t: t**2

# Test at x = 2
print(theorem1(f, 2))
