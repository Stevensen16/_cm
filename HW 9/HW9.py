```python
"""
AI Q&A Tutor (Offline) — z-test & t-test math principles + derivations
--------------------------------------------------------------------
This script is a self-contained “Q&A style” tutor that explains:
- z-test (one-sample, sigma known; mu0 known)
- t-test (one-sample, sigma unknown; mu0 known)
- two-sample independent t-test
- paired t-test

It focuses on the *mathematical reasoning* and *where formulas come from*:
- sampling distributions (Normal)
- standardization
- why Z ~ N(0,1)
- why T has t distribution (using chi-square / sample variance)
- how the two-sample and paired forms reduce to a one-sample t on differences

It also includes:
- compute test statistic + p-value (two-sided / one-sided)
- confidence intervals
- small quiz mode

Requirements:
- Python 3.9+
- No external libraries needed for explanations
- For p-values without SciPy, we implement:
  - Normal CDF via math.erf
  - Student-t CDF via numerical integration (Simpson's rule)
This is accurate enough for learning and homework purposes.

Run:
python zt_tests_tutor.py
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------
# Basic distributions helpers
# ---------------------------

def normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x) using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_sf(x: float) -> float:
    """Standard normal survival function 1-Φ(x)."""
    return 1.0 - normal_cdf(x)


def t_pdf(x: float, df: int) -> float:
    """
    Student's t PDF:
    f(x) = Γ((ν+1)/2) / (sqrt(νπ) Γ(ν/2)) * (1 + x^2/ν)^(-(ν+1)/2)
    """
    nu = df
    c = math.gamma((nu + 1) / 2.0) / (math.sqrt(nu * math.pi) * math.gamma(nu / 2.0))
    return c * (1.0 + (x * x) / nu) ** (-(nu + 1) / 2.0)


def simpson_integral(f, a: float, b: float, n: int = 4000) -> float:
    """
    Numerical integration using Simpson's rule.
    n must be even.
    """
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        s += (4 if i % 2 == 1 else 2) * f(x)
    return s * h / 3.0


def t_cdf(x: float, df: int) -> float:
    """
    Student's t CDF via numeric integration of PDF.
    Uses symmetry: F(-x)=1-F(x)
    """
    if df <= 0:
        raise ValueError("df must be positive.")
    if x == 0.0:
        return 0.5
    if x < 0:
        return 1.0 - t_cdf(-x, df)

    # Integrate from 0 to x then add 0.5 due to symmetry.
    area = simpson_integral(lambda u: t_pdf(u, df), 0.0, x, n=4000)
    return min(1.0, 0.5 + area)


def t_sf(x: float, df: int) -> float:
    return 1.0 - t_cdf(x, df)


def p_value_from_stat(stat: float, dist: str, df: Optional[int], alternative: str) -> float:
    """
    alternative: 'two-sided', 'greater', 'less'
    dist: 'z' or 't'
    """
    alternative = alternative.lower()
    if dist == "z":
        if alternative == "two-sided":
            return 2.0 * min(normal_cdf(stat), normal_sf(stat))
        if alternative == "greater":
            return normal_sf(stat)
        if alternative == "less":
            return normal_cdf(stat)
        raise ValueError("alternative must be: two-sided / greater / less")
    elif dist == "t":
        if df is None:
            raise ValueError("df required for t distribution")
        if alternative == "two-sided":
            return 2.0 * min(t_cdf(stat, df), t_sf(stat, df))
        if alternative == "greater":
            return t_sf(stat, df)
        if alternative == "less":
            return t_cdf(stat, df)
        raise ValueError("alternative must be: two-sided / greater / less")
    else:
        raise ValueError("dist must be 'z' or 't'")


def inv_normal_cdf(p: float) -> float:
    """
    Approximate inverse Φ^{-1}(p) using Acklam's approximation.
    Good accuracy for common CI usage.

    Reference: Peter J. Acklam approximation (widely used).
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    # Coefficients in rational approximations.
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return num / den
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return num / den

    q = p - 0.5
    r = q*q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    return num / den


# NOTE: Inverse t CDF is not implemented (to keep it simple).
# For CI we’ll compute critical t via bisection using t_cdf.


def inv_t_cdf(p: float, df: int) -> float:
    """Inverse CDF for t distribution via bisection."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    if df <= 0:
        raise ValueError("df must be positive")

    # symmetric; for p>0.5 positive, else negative
    if p == 0.5:
        return 0.0

    # bracket
    lo, hi = -50.0, 50.0
    # bisection
    for _ in range(120):
        mid = (lo + hi) / 2.0
        fm = t_cdf(mid, df) - p
        if fm == 0:
            return mid
        if fm < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------
# Core statistics
# ---------------------------

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def sample_variance(xs: List[float]) -> float:
    """Unbiased sample variance s^2 with denominator (n-1)."""
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 observations for sample variance.")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def sample_std(xs: List[float]) -> float:
    return math.sqrt(sample_variance(xs))


@dataclass
class TestResult:
    test_name: str
    statistic: float
    df: Optional[int]
    p_value: float
    alternative: str
    alpha: float
    reject_h0: bool
    ci: Optional[Tuple[float, float]] = None


def z_test_one_sample(
    xbar: float,
    mu0: float,
    sigma: float,
    n: int,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    ci_level: Optional[float] = 0.95
) -> TestResult:
    """
    z-test (one-sample) when population sigma is known.
    Z = (x̄ - μ0) / (σ/√n)
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    se = sigma / math.sqrt(n)
    z = (xbar - mu0) / se
    p = p_value_from_stat(z, dist="z", df=None, alternative=alternative)

    reject = p < alpha

    ci = None
    if ci_level is not None:
        zcrit = inv_normal_cdf(0.5 + ci_level / 2.0)
        ci = (xbar - zcrit * se, xbar + zcrit * se)

    return TestResult(
        test_name="z-test (one-sample, sigma known)",
        statistic=z,
        df=None,
        p_value=p,
        alternative=alternative,
        alpha=alpha,
        reject_h0=reject,
        ci=ci
    )


def t_test_one_sample(
    xs: List[float],
    mu0: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    ci_level: Optional[float] = 0.95
) -> TestResult:
    """
    one-sample t-test when sigma unknown:
    T = (x̄ - μ0) / (s/√n), df=n-1
    """
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 observations for one-sample t-test.")
    xbar = mean(xs)
    s = sample_std(xs)
    se = s / math.sqrt(n)
    t = (xbar - mu0) / se
    df = n - 1
    p = p_value_from_stat(t, dist="t", df=df, alternative=alternative)
    reject = p < alpha

    ci = None
    if ci_level is not None:
        tcrit = inv_t_cdf(0.5 + ci_level / 2.0, df)
        ci = (xbar - tcrit * se, xbar + tcrit * se)

    return TestResult(
        test_name="t-test (one-sample, sigma unknown)",
        statistic=t,
        df=df,
        p_value=p,
        alternative=alternative,
        alpha=alpha,
        reject_h0=reject,
        ci=ci
    )


def t_test_two_sample_independent(
    x: List[float],
    y: List[float],
    equal_var: bool = False,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    ci_level: Optional[float] = 0.95
) -> TestResult:
    """
    Two-sample independent t-test.

    If equal_var=True (pooled variance):
      sp^2 = ((n1-1)s1^2 + (n2-1)s2^2)/(n1+n2-2)
      T = (x̄ - ȳ) / (sp * sqrt(1/n1 + 1/n2)), df=n1+n2-2

    If equal_var=False (Welch):
      T = (x̄ - ȳ) / sqrt(s1^2/n1 + s2^2/n2)
      df via Welch–Satterthwaite approximation
    """
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 observations in each group.")
    xbar, ybar = mean(x), mean(y)
    s1_2 = sample_variance(x)
    s2_2 = sample_variance(y)

    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1_2 + (n2 - 1) * s2_2) / df
        se = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        t = (xbar - ybar) / se
        p = p_value_from_stat(t, dist="t", df=df, alternative=alternative)
        reject = p < alpha

        ci = None
        if ci_level is not None:
            tcrit = inv_t_cdf(0.5 + ci_level / 2.0, df)
            diff = xbar - ybar
            ci = (diff - tcrit * se, diff + tcrit * se)

        return TestResult(
            test_name="t-test (two-sample independent, pooled variance)",
            statistic=t,
            df=df,
            p_value=p,
            alternative=alternative,
            alpha=alpha,
            reject_h0=reject,
            ci=ci
        )

    # Welch
    se2 = s1_2 / n1 + s2_2 / n2
    se = math.sqrt(se2)
    t = (xbar - ybar) / se

    # Welch–Satterthwaite df
    num = se2 ** 2
    den = (s1_2 ** 2) / (n1 ** 2 * (n1 - 1)) + (s2_2 ** 2) / (n2 ** 2 * (n2 - 1))
    df = int(round(num / den))

    p = p_value_from_stat(t, dist="t", df=df, alternative=alternative)
    reject = p < alpha

    ci = None
    if ci_level is not None:
        tcrit = inv_t_cdf(0.5 + ci_level / 2.0, df)
        diff = xbar - ybar
        ci = (diff - tcrit * se, diff + tcrit * se)

    return TestResult(
        test_name="t-test (two-sample independent, Welch)",
        statistic=t,
        df=df,
        p_value=p,
        alternative=alternative,
        alpha=alpha,
        reject_h0=reject,
        ci=ci
    )


def t_test_paired(
    before: List[float],
    after: List[float],
    alternative: str = "two-sided",
    alpha: float = 0.05,
    ci_level: Optional[float] = 0.95
) -> TestResult:
    """
    Paired t-test:
      Define di = before_i - after_i (or after-before, consistent!)
      Then do one-sample t-test on d with H0: μd = 0

      T = d̄ / (sd/√n), df=n-1
    """
    if len(before) != len(after):
        raise ValueError("Paired samples must have the same length.")
    d = [b - a for b, a in zip(before, after)]
    res = t_test_one_sample(d, mu0=0.0, alternative=alternative, alpha=alpha, ci_level=ci_level)
    res.test_name = "t-test (paired)"
    return res


# ---------------------------
# Tutor content (Q&A)
# ---------------------------

DERIVATION_TEXT = r"""
==========================
(1) z-test：為什麼公式是 Z = (x̄ - μ0) / (σ/√n) ?
==========================

核心想法：把「樣本平均 x̄」轉成一個已知分佈的標準化變數，才能算 p-value。

(A) 先看樣本平均的分佈（Sampling Distribution）
假設 X1, X2, ..., Xn 是 i.i.d. 且來自母體：
- E[Xi] = μ
- Var(Xi) = σ^2

定義樣本平均：
    x̄ = (1/n) Σ Xi

期望值（線性性）：
    E[x̄] = E[(1/n) Σ Xi] = (1/n) Σ E[Xi] = (1/n) * nμ = μ

變異數（獨立相加）：
    Var(x̄) = Var((1/n) Σ Xi)
            = (1/n^2) Σ Var(Xi)       (因為獨立，Cov=0)
            = (1/n^2) * nσ^2
            = σ^2 / n

所以標準差（Standard Error）：
    SD(x̄) = σ / √n

(B) 若母體是 Normal，則 x̄ 也是 Normal（精確）
若 Xi ~ N(μ, σ^2)，則
    x̄ ~ N(μ, σ^2/n)

(C) 標準化（Standardize）得到 Z ~ N(0,1)
在 H0: μ = μ0 下，
    x̄ ~ N(μ0, σ^2/n)

定義
    Z = (x̄ - μ0) / (σ/√n)

因為 x̄ 的均值是 μ0、標準差是 σ/√n，所以 Z 會是標準常態：
    Z ~ N(0,1)

(D) 為什麼要「母體標準差 σ 已知」？
因為分母用的是 σ/√n。若 σ 未知，就必須用樣本標準差 s 代替，分佈不再是標準常態，會變成 t 分佈（下面 t-test 會推導）。

補充：若母體不一定 Normal，但 n 大，中央極限定理（CLT）讓 x̄ 近似 Normal，
因此 z-test 在大樣本也能用（但你題目指定「σ 已知」的典型 z-test）。
"""

DERIVATION_T_TEXT = r"""
==========================
(2) one-sample t-test：為什麼 T = (x̄ - μ0) / (s/√n) 而且服從 t 分佈？
==========================

這裡的關鍵：分母不是 σ，而是用樣本標準差 s（隨機變數）。

(A) 假設 Xi ~ N(μ, σ^2)，且獨立同分佈
我們知道：
1) x̄ ~ N(μ, σ^2/n)

2) 樣本方差：
    s^2 = (1/(n-1)) Σ (Xi - x̄)^2

重要結果（正態母體下成立）：
    (n-1)s^2 / σ^2 ~ χ^2_{n-1}   （卡方分佈）

(B) 而且 x̄ 與 s^2 在正態母體下是獨立的（很關鍵）

(C) 把 Z 與 χ^2 組合成 t 分佈
先定義（在 H0: μ=μ0 下）：
    Z = (x̄ - μ0) / (σ/√n) ~ N(0,1)

以及
    U = (n-1)s^2 / σ^2 ~ χ^2_{n-1}

且 Z ⟂ U（獨立）

t 分佈的定義就是：
    T = Z / sqrt(U/(n-1))  ~ t_{n-1}

把 Z、U 代回去：
    T = [(x̄ - μ0)/(σ/√n)] / sqrt( [(n-1)s^2/σ^2]/(n-1) )
      = [(x̄ - μ0)/(σ/√n)] / sqrt( s^2/σ^2 )
      = (x̄ - μ0) / (s/√n)

所以：
    T = (x̄ - μ0) / (s/√n)  ~ t_{n-1}

(D) 直覺：為什麼會「尾巴更厚」？
因為 s 是從資料估出來的，存在估計不確定性，
所以 T 的變動比 Z 大，尾巴更厚 → 用 t 分佈補償不確定性。
df 越大（樣本越多），s 越穩定，t 分佈越接近常態。
"""

DERIVATION_TWO_SAMPLE_TEXT = r"""
==========================
(3) 兩獨立樣本 t-test：公式怎麼來？
==========================

目標：檢定 H0: μ1 - μ2 = 0（或指定差異值）
資料：
- 第一組：X1..Xn1 ~ N(μ1, σ1^2)
- 第二組：Y1..Yn2 ~ N(μ2, σ2^2)
且兩組彼此獨立。

(A) 先看差的平均：D = x̄ - ȳ
因為 x̄ 與 ȳ 獨立：
    E[D] = E[x̄] - E[ȳ] = μ1 - μ2
    Var(D) = Var(x̄) + Var(ȳ)
           = σ1^2/n1 + σ2^2/n2

所以標準誤：
    SE = sqrt( σ1^2/n1 + σ2^2/n2 )

如果 σ1, σ2 已知，類似 z-test：
    Z = ( (x̄ - ȳ) - (μ1-μ2)_0 ) / SE

(B) 但通常 σ1, σ2 未知 → 用 s1^2, s2^2 代替
Welch t-test（不假設等變異）：
    T = (x̄ - ȳ) / sqrt( s1^2/n1 + s2^2/n2 )

df 用 Welch–Satterthwaite 近似：
    df ≈ (s1^2/n1 + s2^2/n2)^2
          / [ (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) ]

(C) 若額外假設等變異（σ1^2 = σ2^2 = σ^2）
可做 pooled variance：
    sp^2 = [ (n1-1)s1^2 + (n2-1)s2^2 ] / (n1+n2-2)

則
    T = (x̄ - ȳ) / ( sp * sqrt(1/n1 + 1/n2) )
df = n1 + n2 - 2
"""

DERIVATION_PAIRED_TEXT = r"""
==========================
(4) 配對樣本 t-test：為什麼可以變成 one-sample t-test？
==========================

配對樣本：同一個人/物件在兩時間點（或前後測）
例如：
- before_i
- after_i

重點：兩次測量不是獨立的（同一人），所以不能用獨立樣本 t-test。

做法：把配對差值當成一組新的樣本
定義：
    di = before_i - after_i   （或 after-before，都可以，只要一致）

如果 H0: 兩時間點平均無差異
等價於：
    H0: μd = 0

那就是對 d1..dn 做 one-sample t-test：
    T = d̄ / (sd/√n), df = n-1

直覺：配對能消掉「個體差異」
因為同一人前後差值中，很多固定的個體特徵會抵消，
讓變異變小，檢定力通常更強。
"""


# ---------------------------
# UI / Tutor interactions
# ---------------------------

def print_menu():
    print("\n=== z-test & t-test AI Q&A Tutor (Offline) ===")
    print("1) Explain derivation: z-test (one-sample)")
    print("2) Explain derivation: t-test (one-sample)")
    print("3) Explain derivation: independent two-sample t-test (Welch/pooled)")
    print("4) Explain derivation: paired t-test")
    print("5) Calculate: z-test (one-sample, sigma known)")
    print("6) Calculate: t-test (one-sample)")
    print("7) Calculate: t-test (two-sample independent)")
    print("8) Calculate: t-test (paired)")
    print("9) Mini quiz (concept check)")
    print("0) Exit")


def ask_alt_alpha() -> Tuple[str, float]:
    alt = input("Alternative (two-sided / greater / less) [two-sided]: ").strip().lower() or "two-sided"
    alpha_raw = input("alpha [0.05]: ").strip()
    alpha = float(alpha_raw) if alpha_raw else 0.05
    return alt, alpha


def format_result(res: TestResult) -> str:
    lines = []
    lines.append(f"Test: {res.test_name}")
    if res.df is None:
        lines.append(f"Statistic: {res.statistic:.6f}")
    else:
        lines.append(f"Statistic: {res.statistic:.6f}   df={res.df}")
    lines.append(f"Alternative: {res.alternative}   alpha={res.alpha}")
    lines.append(f"p-value: {res.p_value:.6g}")
    lines.append(f"Decision: {'REJECT H0' if res.reject_h0 else 'FAIL TO REJECT H0'}")
    if res.ci is not None:
        lo, hi = res.ci
        lines.append(f"CI: [{lo:.6f}, {hi:.6f}]")
    return "\n".join(lines)


def parse_floats(prompt: str) -> List[float]:
    """
    Input format:
      1 2 3 4
    or:
      1,2,3,4
    """
    raw = input(prompt).strip()
    raw = raw.replace(",", " ")
    xs = [float(tok) for tok in raw.split() if tok]
    if not xs:
        raise ValueError("No numbers provided.")
    return xs


def mini_quiz():
    questions = [
        ("z-test 用到的分母 σ/√n 代表什麼？", "樣本平均的標準誤（standard error）"),
        ("one-sample t-test 為什麼不是用常態而是用 t 分佈？", "因為 σ 未知用 s 代替，分母變成隨機；Z 與 χ² 組合得到 t"),
        ("配對 t-test 的核心轉換是什麼？", "把 before-after 的差值當作新樣本，做一樣本 t 檢定"),
        ("Welch t-test 主要解決什麼問題？", "兩組變異數不等或不想假設等變異"),
    ]
    print("\n--- Mini Quiz ---")
    score = 0
    for i, (q, key) in enumerate(questions, 1):
        print(f"\nQ{i}: {q}")
        ans = input("Your answer (free text): ").strip()
        print(f"Key idea: {key}")
        if ans:
            score += 1
    print(f"\nDone. You attempted {score}/{len(questions)} questions.")


def main():
    while True:
        print_menu()
        choice = input("Select: ").strip()

        if choice == "0":
            print("Bye.")
            return

        if choice == "1":
            print(DERIVATION_TEXT)
        elif choice == "2":
            print(DERIVATION_T_TEXT)
        elif choice == "3":
            print(DERIVATION_TWO_SAMPLE_TEXT)
        elif choice == "4":
            print(DERIVATION_PAIRED_TEXT)

        elif choice == "5":
            print("\n--- z-test (one-sample, sigma known) ---")
            xbar = float(input("x̄ (sample mean): ").strip())
            mu0 = float(input("μ0 (null mean): ").strip())
            sigma = float(input("σ (population std, known): ").strip())
            n = int(input("n (sample size): ").strip())
            alt, alpha = ask_alt_alpha()
            res = z_test_one_sample(xbar=xbar, mu0=mu0, sigma=sigma, n=n, alternative=alt, alpha=alpha)
            print("\n" + format_result(res))

        elif choice == "6":
            print("\n--- t-test (one-sample) ---")
            xs = parse_floats("Enter sample values (space or comma separated): ")
            mu0 = float(input("μ0 (null mean): ").strip())
            alt, alpha = ask_alt_alpha()
            res = t_test_one_sample(xs=xs, mu0=mu0, alternative=alt, alpha=alpha)
            print("\n" + format_result(res))

        elif choice == "7":
            print("\n--- t-test (two-sample independent) ---")
            x = parse_floats("Enter group X values: ")
            y = parse_floats("Enter group Y values: ")
            ev = input("Assume equal variances? (y/n) [n]: ").strip().lower() or "n"
            equal_var = (ev == "y")
            alt, alpha = ask_alt_alpha()
            res = t_test_two_sample_independent(x=x, y=y, equal_var=equal_var, alternative=alt, alpha=alpha)
            print("\n" + format_result(res))

        elif choice == "8":
            print("\n--- t-test (paired) ---")
            before = parse_floats("Enter BEFORE values: ")
            after = parse_floats("Enter AFTER values: ")
            alt, alpha = ask_alt_alpha()
            res = t_test_paired(before=before, after=after, alternative=alt, alpha=alpha)
            print("\n" + format_result(res))

        elif choice == "9":
            mini_quiz()

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
```
