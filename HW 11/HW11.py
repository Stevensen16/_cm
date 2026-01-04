import numpy as np

# =========================================================
# Continuous Fourier Transform (numerical approximation)
#   F(w) = ∫ f(x) e^{-i w x} dx
# Inverse:
#   f(x) = (1/2π) ∫ F(w) e^{i w x} dw
#
# We approximate integrals with discrete sums:
#   F(w_k) ≈ Σ_n f(x_n) e^{-i w_k x_n} dx
#   f(x_n) ≈ (1/2π) Σ_k F(w_k) e^{i w_k x_n} dw
#
# We choose w grid consistent with x grid:
#   dx = x[1]-x[0], N = len(x), L = N*dx
#   w_k = 2π * k / L  (k = -N/2 .. N/2-1)
#   dw = 2π / L
# =========================================================

def make_omega_grid(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Given uniform x grid, return omega grid (shifted) and dw."""
    x = np.asarray(x, dtype=float)
    N = x.size
    dx = x[1] - x[0]
    L = N * dx
    dw = 2.0 * np.pi / L

    # k indices centered at 0: [-N/2, ..., N/2-1]
    k = np.arange(N) - (N // 2)
    w = dw * k
    return w, dw

def dft(fx: np.ndarray, x: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerical Fourier transform:
      F(w) ≈ Σ f(x) e^{-i w x} dx
    Returns (F, w).
    """
    fx = np.asarray(fx, dtype=complex)
    x = np.asarray(x, dtype=float)
    N = x.size
    dx = x[1] - x[0]

    if w is None:
        w, _ = make_omega_grid(x)
    w = np.asarray(w, dtype=float)

    # Build matrix exp(-i w x): shape (Nw, Nx)
    E = np.exp(-1j * np.outer(w, x))  # (len(w), N)
    Fw = E @ fx * dx
    return Fw, w

def idft(Fw: np.ndarray, w: np.ndarray, x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerical inverse Fourier transform:
      f(x) ≈ (1/2π) Σ F(w) e^{i w x} dw
    Returns (f_rec, x).
    """
    Fw = np.asarray(Fw, dtype=complex)
    w = np.asarray(w, dtype=float)
    Nw = w.size
    dw = w[1] - w[0]

    if x is None:
        # If x not provided, construct x-grid consistent with w-grid:
        # dw = 2π / L => L = 2π / dw, dx = L / N
        L = 2.0 * np.pi / dw
        dx = L / Nw
        x = (np.arange(Nw) - (Nw // 2)) * dx
    x = np.asarray(x, dtype=float)

    E = np.exp(1j * np.outer(x, w))  # (Nx, Nw)
    fx_rec = (E @ Fw) * (dw / (2.0 * np.pi))
    return fx_rec, x

# =========================================================
# 3) Verification demo
# =========================================================

def demo_verify_gaussian(N=256, x_max=10.0):
    """
    Use Gaussian: f(x)=exp(-x^2/2)
    True FT (with this convention): F(w)=sqrt(2π)*exp(-w^2/2)
    Then invert and compare.
    """
    x = np.linspace(-x_max, x_max, N, endpoint=False)
    fx = np.exp(-0.5 * x**2)

    # Forward transform
    Fw, w = dft(fx, x)

    # Inverse transform
    fx_rec, _ = idft(Fw, w, x=x)

    # Error metrics
    fx_rec_real = fx_rec.real
    max_abs_err = np.max(np.abs(fx_rec_real - fx))
    rmse = np.sqrt(np.mean((fx_rec_real - fx) ** 2))

    print("=== Gaussian verification ===")
    print(f"N={N}, x in [{x[0]:.2f}, {x[-1]:.2f}], dx={x[1]-x[0]:.6f}")
    print(f"max |f_rec - f| = {max_abs_err:.3e}")
    print(f"RMSE            = {rmse:.3e}")

    # Optional: compare with analytic transform (for learning)
    F_true = np.sqrt(2.0 * np.pi) * np.exp(-0.5 * w**2)
    max_F_err = np.max(np.abs(Fw.real - F_true))  # F should be (almost) real & even
    print(f"max |Re(F_num) - F_true| = {max_F_err:.3e}")

def demo_verify_cosine(N=256, x_max=10.0, w0=2.0):
    """
    Another function: f(x)=cos(w0 x) * exp(-x^2/10)
    It's band-limited-ish and decays, good for numeric FT.
    """
    x = np.linspace(-x_max, x_max, N, endpoint=False)
    fx = np.cos(w0 * x) * np.exp(-(x**2) / 10.0)

    Fw, w = dft(fx, x)
    fx_rec, _ = idft(Fw, w, x=x)

    fx_rec_real = fx_rec.real
    max_abs_err = np.max(np.abs(fx_rec_real - fx))
    rmse = np.sqrt(np.mean((fx_rec_real - fx) ** 2))

    print("\n=== Cosine*Gaussian verification ===")
    print(f"N={N}, w0={w0}")
    print(f"max |f_rec - f| = {max_abs_err:.3e}")
    print(f"RMSE            = {rmse:.3e}")

if __name__ == "__main__":
    demo_verify_gaussian(N=256, x_max=10.0)
    demo_verify_cosine(N=256, x_max=10.0, w0=2.0)
