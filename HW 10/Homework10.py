import numpy as np

# ============================================================
# A) CONCEPT NOTES (printed output)
# ============================================================

def concept_notes() -> str:
    return (
        "==================== 概念題（重點版）====================\n"
        "\n"
        "1) 線性代數中的『線性』是什麼？為何稱為『代數』？\n"
        "   - 線性（linear）指的是滿足『可加性』與『齊次性』的關係：\n"
        "       f(x+y)=f(x)+f(y),  f(cx)=c f(x)\n"
        "     在矩陣表示下就是 y = A x（或 y = A x + b，後者是仿射）。\n"
        "   - 為何叫代數（algebra）：因為研究的核心是用符號/運算規則來操控\n"
        "     向量、矩陣、線性映射，包含加法、乘法、逆、分解等『運算結構』。\n"
        "\n"
        "2) 數學中的『空間』是什麼？為何『向量空間』叫空間？\n"
        "   - 數學空間：一個集合 + 結構（運算、距離、內積、拓樸…）。\n"
        "   - 向量空間：一個集合 V + 兩種運算（向量加法、純量乘法）並滿足公理。\n"
        "     叫『空間』是因為它抽象化了我們熟悉的幾何空間概念（可線性組合、\n"
        "     可定義方向/維度/基底）。\n"
        "\n"
        "3) 矩陣和向量的關係？矩陣代表什麼？\n"
        "   - 向量是元素的集合（可視為座標）；矩陣是線性映射的表示（把向量映到向量）。\n"
        "     y = A x 表示：A 對 x 做一個線性變換。\n"
        "   - A 的每一欄（column）可視為基底向量被映射後的位置：\n"
        "     A = [A e1, A e2, ..., A en]\n"
        "\n"
        "4) 用矩陣表示 2D/3D 幾何：平移、縮放、旋轉\n"
        "   - 縮放（2D）：S = [[sx,0],[0,sy]]，x' = Sx\n"
        "   - 旋轉（2D）：R(θ) = [[cosθ,-sinθ],[sinθ,cosθ]]\n"
        "   - 平移不是線性（因為不滿足 f(0)=0），通常用『齊次座標 homogeneous coordinates』\n"
        "     把 2D 變成 3D：\n"
        "       [x' y' 1]^T = [[1,0,tx],[0,1,ty],[0,0,1]] [x y 1]^T\n"
        "     3D 平移用 4x4 齊次矩陣。\n"
        "\n"
        "5) 行列式意義？遞迴公式？與體積關係？\n"
        "   - det(A) 表示線性映射的『體積縮放因子』與『方向（正負）』。\n"
        "     |det(A)|：n 維體積縮放倍率；det(A)<0 代表翻轉。\n"
        "   - 遞迴（Laplace 展開）：\n"
        "       det(A) = Σ_j (-1)^{i+j} a_{i,j} det(M_{i,j})\n"
        "     M_{i,j} 是去掉第 i 列第 j 欄的子矩陣。\n"
        "\n"
        "6) 透過對角化快速算 det\n"
        "   - 若 A 可對角化：A = P D P^{-1}，則 det(A)=det(D)=∏ λ_i\n"
        "     因為 det(P)det(D)det(P^{-1})=det(D)。\n"
        "\n"
        "7) 用 LU 分解快速算 det\n"
        "   - 若 PA = LU（帶 pivot），則 det(A)=det(P^{-1}) det(L) det(U)\n"
        "     det(L)=1（若 L 對角線為 1），det(U)=∏ U_ii\n"
        "     det(P^{-1}) = det(P) = (-1)^{#swaps}\n"
        "\n"
        "8) 特徵值/特徵向量意義？用途？\n"
        "   - Av = λv：v 的方向在變換後不變，只被縮放 λ。\n"
        "   - 用途：動態系統、穩定性、PCA（協方差特徵向量）、矩陣函數 e^A、\n"
        "     快速計算 det、trace、對角化/近似等。\n"
        "\n"
        "9) QR 分解是什麼？\n"
        "   - A = QR，其中 Q 正交（Q^TQ=I），R 上三角。\n"
        "   - 用途：最小平方、數值穩定的正交化、特徵值算法（QR iteration）。\n"
        "\n"
        "10) 反覆用 QR 完成特徵值分解（QR iteration）\n"
        "   - 迭代：A_k = Q_k R_k，A_{k+1} = R_k Q_k\n"
        "   - 在條件適當時，A_k 會趨近上三角，其對角線趨近特徵值。\n"
        "     若累乘 Q_total = Q0 Q1 ...，可得到特徵向量近似。\n"
        "\n"
        "11) SVD 是什麼？和特徵值分解關係？\n"
        "   - A = U Σ V^T\n"
        "     U,V 正交；Σ 對角非負（奇異值）。\n"
        "   - 關係：A^T A 的特徵值 = σ_i^2，特徵向量 = V\n"
        "            A A^T 的特徵向量 = U\n"
        "\n"
        "12) PCA 是什麼？和 SVD 關係？\n"
        "   - PCA 找資料最大變異方向（主成分）。\n"
        "   - 對中心化資料矩陣 X（n_samples x n_features）：\n"
        "       X = U Σ V^T\n"
        "     主成分方向 = V；解釋變異 = (Σ^2)/(n-1)。\n"
        "\n"
        "==========================================================\n"
    )


# ============================================================
# B) 2D/3D GEOMETRY MATRICES (demo utilities)
# ============================================================

def rot2d(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def scale2d(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0],
                     [0.0, sy]], dtype=float)

def translate2d_hom(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx],
                     [0.0, 1.0, ty],
                     [0.0, 0.0, 1.0]], dtype=float)


# ============================================================
# C) DETERMINANT (Recursive Laplace Expansion)
# ============================================================

def det_recursive(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("det_recursive requires a square matrix.")
    if n == 1:
        return float(A[0, 0])
    if n == 2:
        return float(A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0])

    # Expand along first row (i=0)
    total = 0.0
    for j in range(n):
        if A[0, j] == 0:
            continue
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        cofactor = ((-1) ** j) * A[0, j] * det_recursive(minor)
        total += cofactor
    return float(total)


# ============================================================
# D) LU Decomposition (with partial pivoting): PA = LU
# ============================================================

def lu_decomposition_pp(A: np.ndarray):
    """
    Partial pivoting LU: returns P, L, U, num_swaps
    such that P @ A = L @ U
    L has unit diagonal.
    """
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("LU requires square matrix.")

    U = A.copy()
    L = np.eye(n, dtype=float)
    P = np.eye(n, dtype=float)
    num_swaps = 0

    for k in range(n - 1):
        # pivot row
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[pivot, k]) < 1e-15:
            # singular or near-singular
            continue
        if pivot != k:
            # swap in U
            U[[k, pivot], :] = U[[pivot, k], :]
            # swap in P
            P[[k, pivot], :] = P[[pivot, k], :]
            # swap in L for columns < k
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
            num_swaps += 1

        # elimination
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]

    return P, L, U, num_swaps

def det_via_lu(A: np.ndarray) -> float:
    P, L, U, swaps = lu_decomposition_pp(A)
    detP = -1.0 if (swaps % 2 == 1) else 1.0
    detU = float(np.prod(np.diag(U)))
    detL = 1.0  # unit diagonal
    # From P A = L U => det(P) det(A) = det(L) det(U)
    # => det(A) = det(L) det(U) / det(P)
    # det(P) = ±1, and det(P^{-1}) = det(P)
    return float((detL * detU) / detP)


# ============================================================
# E) Verify factorizations: LU / Eig / SVD
# ============================================================

def verify_lu(A: np.ndarray, atol=1e-8) -> bool:
    P, L, U, _ = lu_decomposition_pp(A)
    left = P @ A
    right = L @ U
    return np.allclose(left, right, atol=atol)

def verify_eigendecomp(A: np.ndarray, atol=1e-8) -> bool:
    """
    Works best for diagonalizable matrices. For symmetric matrices it's stable.
    We'll test with a symmetric matrix in demo.
    """
    vals, vecs = np.linalg.eig(A)
    A_rec = vecs @ np.diag(vals) @ np.linalg.inv(vecs)
    return np.allclose(A, A_rec, atol=atol)

def verify_svd(A: np.ndarray, atol=1e-8) -> bool:
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    A_rec = U @ np.diag(s) @ Vt
    return np.allclose(A, A_rec, atol=atol)


# ============================================================
# F) Build SVD from eigen-decomposition
# ============================================================

def svd_from_eigendecomp(A: np.ndarray, eps=1e-12):
    """
    Compute SVD using eigen-decomposition of A^T A:
      A^T A = V diag(s^2) V^T  (for real symmetric)
    Then:
      sigma_i = sqrt(eigenvalues)
      U_i = (A v_i) / sigma_i   for sigma_i > 0
    """
    A = np.asarray(A, dtype=float)
    AtA = A.T @ A

    # AtA is symmetric PSD, use eigh (stable, real)
    evals, V = np.linalg.eigh(AtA)

    # sort descending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    V = V[:, idx]

    sigmas = np.sqrt(np.clip(evals, 0.0, None))

    # build U
    U_cols = []
    s_used = []
    V_cols = []

    for i, sigma in enumerate(sigmas):
        if sigma > eps:
            v = V[:, i]
            u = (A @ v) / sigma
            U_cols.append(u)
            s_used.append(sigma)
            V_cols.append(v)

    if not U_cols:
        # A is (close to) zero matrix
        m, n = A.shape
        U = np.eye(m, dtype=float)
        Vt = np.eye(n, dtype=float)
        s = np.zeros(min(m, n), dtype=float)
        return U[:, :len(s)], s, Vt[:len(s), :]

    U = np.column_stack(U_cols)
    s = np.array(s_used, dtype=float)
    V = np.column_stack(V_cols)
    Vt = V.T

    # Orthonormalize U (numerical cleanup)
    # QR gives U_orth and adjusts, but keep simple:
    U, _ = np.linalg.qr(U)

    return U, s, Vt


# ============================================================
# G) PCA using SVD
# ============================================================

def pca_svd(X: np.ndarray, k: int):
    """
    PCA via SVD on centered data X (n_samples x n_features)
    Returns:
      components (k x n_features)
      explained_variance (k,)
      explained_variance_ratio (k,)
      projected_data (n_samples x k)
      mean (n_features,)
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if k < 1 or k > d:
        raise ValueError("k must be between 1 and number of features.")

    mu = X.mean(axis=0)
    Xc = X - mu

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    # eigenvalues of covariance = s^2/(n-1)
    explained_variance = (s**2) / max(n - 1, 1)
    total_var = explained_variance.sum() if explained_variance.sum() > 0 else 1.0
    explained_variance_ratio = explained_variance / total_var

    components = Vt[:k, :]           # rows are principal directions
    projected = Xc @ components.T    # n x k

    return components, explained_variance[:k], explained_variance_ratio[:k], projected, mu


# ============================================================
# H) QR Iteration demo (eigenvalues approximation)
# ============================================================

def qr_iteration_eigenvalues(A: np.ndarray, iters: int = 50):
    """
    Basic QR iteration (no shifts) to approximate eigenvalues.
    Best for symmetric matrices.
    """
    Ak = np.asarray(A, dtype=float)
    for _ in range(iters):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
    return np.diag(Ak)


# ============================================================
# I) MAIN DEMO
# ============================================================

def main():
    print(concept_notes())

    # ------------------ Determinant demos ------------------
    A = np.array([[2, 1, 3],
                  [0, 1, 4],
                  [5, 2, 0]], dtype=float)

    print("=== 1) 遞迴計算 det(A) ===")
    det_rec = det_recursive(A)
    det_np = float(np.linalg.det(A))
    print("A=\n", A)
    print("det_recursive(A) =", det_rec)
    print("np.linalg.det(A) =", det_np)
    print("close? ", np.isclose(det_rec, det_np, atol=1e-6))
    print()

    print("=== 2) LU 分解計算 det(A) ===")
    det_lu = det_via_lu(A)
    print("det_via_lu(A) =", det_lu)
    print("np.linalg.det(A) =", det_np)
    print("close? ", np.isclose(det_lu, det_np, atol=1e-6))
    print("verify P@A=L@U ?", verify_lu(A))
    print()

    # ------------------ Verify decompositions ------------------
    print("=== 3) 驗證分解後可重建原矩陣 ===")
    print("SVD reconstruct ok? ", verify_svd(A))
    print("LU reconstruct ok?  ", verify_lu(A))

    # Eigen reconstruction works best for symmetric; use a symmetric matrix:
    S = np.array([[4, 1, 1],
                  [1, 3, 0],
                  [1, 0, 2]], dtype=float)
    print("\nSymmetric S=\n", S)
    print("Eigen reconstruct ok? ", verify_eigendecomp(S))
    print()

    # ------------------ SVD from eigen ------------------
    print("=== 4) 用特徵值分解做 SVD（從 A^T A）===")
    U2, s2, Vt2 = svd_from_eigendecomp(A)
    A_rec2 = U2 @ np.diag(s2) @ Vt2
    print("Singular values (from eig):", s2)
    print("Reconstruct close? ", np.allclose(A, A_rec2, atol=1e-6))
    print()

    # ------------------ PCA demo ------------------
    print("=== 5) PCA 主成分分析（SVD）===")
    rng = np.random.default_rng(0)
    # synthetic data: 200 samples, 3 features with correlation
    X = rng.normal(size=(200, 3))
    X[:, 2] = 0.7 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * rng.normal(size=200)

    k = 2
    comps, ev, evr, Z, mu = pca_svd(X, k=k)
    print("Mean (mu):", mu)
    print("Top-k components (rows):\n", comps)
    print("Explained variance:", ev)
    print("Explained variance ratio:", evr)
    print("Projected data shape:", Z.shape)
    print()

    # ------------------ QR iteration demo ------------------
    print("=== 6) QR 反覆分解近似特徵值（對稱矩陣示範）===")
    approx_eigs = qr_iteration_eigenvalues(S, iters=60)
    true_eigs = np.linalg.eigvalsh(S)
    print("QR approx eigs (diag):", np.sort(approx_eigs))
    print("True eigs:", np.sort(true_eigs))


if __name__ == "__main__":
    main()
