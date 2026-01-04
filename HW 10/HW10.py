import math
from decimal import Decimal, getcontext
from typing import List, Tuple

# =========================
# 1) Fair coin: P(all heads in 10000 flips) = (0.5)^10000
# =========================

def prob_all_heads_float(n: int = 10000, p: float = 0.5) -> float:
    # WARNING: this will underflow to 0.0 in IEEE float for large n
    return p ** n

def prob_all_heads_decimal(n: int = 10000, p: str = "0.5", prec: int = 200) -> Decimal:
    # Use high precision decimal to avoid underflow
    getcontext().prec = prec
    p_dec = Decimal(p)
    return p_dec ** Decimal(n)

def log_prob_power(n: int = 10000, p: float = 0.5, base: str = "e") -> float:
    """
    2) Compute log(p^n) = n log(p)
    base: 'e' for ln, '2' for log2, '10' for log10
    """
    if p <= 0:
        raise ValueError("p must be > 0 for log.")
    if base == "e":
        return n * math.log(p)
    if base == "2":
        return n * math.log2(p)
    if base == "10":
        return n * math.log10(p)
    raise ValueError("base must be one of: 'e', '2', '10'.")

# =========================
# 3) Entropy, Cross-Entropy, KL Divergence, Mutual Information
# =========================

def _normalize(dist: List[float]) -> List[float]:
    s = sum(dist)
    if s <= 0:
        raise ValueError("Distribution sum must be > 0.")
    return [x / s for x in dist]

def entropy(p: List[float], base: float = 2.0) -> float:
    """
    H(p) = - Σ p_i log(p_i)
    base=2 => bits
    """
    p = _normalize(p)
    H = 0.0
    for pi in p:
        if pi > 0:
            H -= pi * (math.log(pi) / math.log(base))
    return H

def cross_entropy(p: List[float], q: List[float], base: float = 2.0, eps: float = 1e-12) -> float:
    """
    H(p, q) = - Σ p_i log(q_i)
    Need q_i > 0 when p_i > 0, otherwise infinite.
    Here we clamp q_i by eps for numeric stability (learning use).
    """
    p = _normalize(p)
    q = _normalize(q)

    H = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            qi_safe = max(qi, eps)
            H -= pi * (math.log(qi_safe) / math.log(base))
    return H

def kl_divergence(p: List[float], q: List[float], base: float = 2.0, eps: float = 1e-12) -> float:
    """
    KL(p||q) = Σ p_i log(p_i / q_i)
    (>=0)
    """
    p = _normalize(p)
    q = _normalize(q)

    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            qi_safe = max(qi, eps)
            kl += pi * (math.log(pi / qi_safe) / math.log(base))
    return kl

def mutual_information(joint: List[List[float]], base: float = 2.0) -> float:
    """
    I(X;Y) = Σ_{x,y} p(x,y) log( p(x,y) / (p(x)p(y)) )
    joint is a matrix p[x][y]
    """
    # Normalize joint
    total = sum(sum(row) for row in joint)
    if total <= 0:
        raise ValueError("Joint distribution sum must be > 0.")
    Pxy = [[v / total for v in row] for row in joint]

    # Marginals
    Px = [sum(row) for row in Pxy]
    Py = [sum(Pxy[i][j] for i in range(len(Pxy))) for j in range(len(Pxy[0]))]

    I = 0.0
    for i in range(len(Pxy)):
        for j in range(len(Pxy[0])):
            pxy = Pxy[i][j]
            if pxy > 0 and Px[i] > 0 and Py[j] > 0:
                I += pxy * (math.log(pxy / (Px[i] * Py[j])) / math.log(base))
    return I

# =========================
# 4) Verify inequality: H(p,p) <= H(p,q) for q != p
# =========================

def verify_cross_entropy_property(p: List[float], q: List[float]) -> Tuple[float, float, float]:
    """
    Returns Hpp, Hpq, KL(p||q).
    Theory: H(p,q) = H(p) + KL(p||q) >= H(p) = H(p,p)
    """
    Hpp = cross_entropy(p, p)
    Hpq = cross_entropy(p, q)
    KL = kl_divergence(p, q)
    return Hpp, Hpq, KL

# =========================
# 5) (7,4) Hamming code: encode & decode
# =========================
# Using standard Hamming(7,4) with parity bits at positions 1,2,4 (1-indexed)
# Positions: 1 p1, 2 p2, 3 d1, 4 p4, 5 d2, 6 d3, 7 d4
#
# Parity check sets:
# p1 covers positions with binary index bit1=1: 1,3,5,7
# p2 covers bit2=1: 2,3,6,7
# p4 covers bit3=1: 4,5,6,7

def hamming74_encode(data_bits: List[int]) -> List[int]:
    """
    data_bits: [d1,d2,d3,d4] each 0/1
    returns 7-bit codeword list positions 1..7 mapped to list index 0..6
    """
    if len(data_bits) != 4 or any(b not in (0,1) for b in data_bits):
        raise ValueError("data_bits must be length 4, bits 0/1")

    d1, d2, d3, d4 = data_bits

    # Place data bits
    c = [0]*7
    c[2] = d1  # pos3
    c[4] = d2  # pos5
    c[5] = d3  # pos6
    c[6] = d4  # pos7

    # Compute parity bits (even parity)
    # p1 covers 1,3,5,7 => positions 1,3,5,7 => indices 0,2,4,6
    p1 = (c[2] + c[4] + c[6]) % 2
    # p2 covers 2,3,6,7 => indices 1,2,5,6
    p2 = (c[2] + c[5] + c[6]) % 2
    # p4 covers 4,5,6,7 => indices 3,4,5,6
    p4 = (c[4] + c[5] + c[6]) % 2

    c[0] = p1
    c[1] = p2
    c[3] = p4
    return c

def hamming74_syndrome(codeword: List[int]) -> int:
    """
    Compute syndrome (1..7 indicates error position, 0 means no error)
    syndrome bits: s1 for parity p1, s2 for p2, s4 for p4
    return integer s = s1*1 + s2*2 + s4*4
    """
    if len(codeword) != 7 or any(b not in (0,1) for b in codeword):
        raise ValueError("codeword must be length 7, bits 0/1")

    c = codeword
    # s1 checks positions 1,3,5,7 (even parity)
    s1 = (c[0] + c[2] + c[4] + c[6]) % 2
    # s2 checks positions 2,3,6,7
    s2 = (c[1] + c[2] + c[5] + c[6]) % 2
    # s4 checks positions 4,5,6,7
    s4 = (c[3] + c[4] + c[5] + c[6]) % 2

    return s1*1 + s2*2 + s4*4

def hamming74_decode(codeword: List[int]) -> Tuple[List[int], List[int], int]:
    """
    Returns: (decoded_data_bits, corrected_codeword, error_position)
    error_position = 0 means no error detected; 1..7 means corrected that bit.
    """
    c = codeword[:]
    s = hamming74_syndrome(c)
    if s != 0:
        # correct the bit at position s (1-indexed)
        idx = s - 1
        c[idx] ^= 1

    # Extract data bits positions 3,5,6,7 => indices 2,4,5,6
    data = [c[2], c[4], c[5], c[6]]
    return data, c, s

# =========================
# Demo / Main
# =========================

def main():
    print("========== 1) Fair coin all heads ==========")
    n = 10000
    p = 0.5
    prob_f = prob_all_heads_float(n, p)
    print(f"(float) P(all heads) = 0.5^{n} = {prob_f}   <-- likely underflow to 0.0")

    prob_d = prob_all_heads_decimal(n, "0.5", prec=250)
    print(f"(Decimal, prec=250) P(all heads) = 0.5^{n} = {prob_d}")
    print(f"Scientific-ish: {prob_d:.5E}")

    print("\n========== 2) log(p^n) = n log p ==========")
    ln_val = log_prob_power(n, p, base="e")
    lg2_val = log_prob_power(n, p, base="2")
    lg10_val = log_prob_power(n, p, base="10")
    print(f"ln(0.5^{n})   = {ln_val}")
    print(f"log2(0.5^{n}) = {lg2_val}  (should be -10000 because log2(0.5)=-1)")
    print(f"log10(0.5^{n})= {lg10_val}")

    print("\n========== 3) Entropy / Cross-Entropy / KL / Mutual Information ==========")
    p_dist = [0.1, 0.2, 0.3, 0.4]
    q_dist = [0.25, 0.25, 0.25, 0.25]
    print(f"p = {p_dist}")
    print(f"q = {q_dist}")
    print(f"H(p)       = {entropy(p_dist):.6f} bits")
    print(f"H(p,q)     = {cross_entropy(p_dist, q_dist):.6f} bits")
    print(f"KL(p||q)   = {kl_divergence(p_dist, q_dist):.6f} bits")
    print("Check identity: H(p,q) = H(p) + KL(p||q)")
    print(f"  H(p)+KL  = {entropy(p_dist)+kl_divergence(p_dist,q_dist):.6f} bits")

    # Mutual information example (joint distribution)
    joint = [
        [0.10, 0.10],
        [0.10, 0.70],
    ]  # sums to 1.0
    print(f"I(X;Y) from joint = {mutual_information(joint):.6f} bits")

    print("\n========== 4) Verify cross-entropy property ==========")
    Hpp, Hpq, KL = verify_cross_entropy_property(p_dist, q_dist)
    print(f"H(p,p) = {Hpp:.6f}")
    print(f"H(p,q) = {Hpq:.6f}")
    print(f"KL(p||q)= {KL:.6f}")
    if Hpp <= Hpq:
        print("✅ Verified: H(p,p) <= H(p,q) (q != p usually makes it strictly smaller).")
    else:
        print("❌ Something off (check q has zeros where p>0, or numeric eps).")

    print("\n========== 5) Hamming(7,4) encode/decode ==========")
    data = [1, 0, 1, 1]  # d1 d2 d3 d4
    code = hamming74_encode(data)
    print(f"Data bits: {data}")
    print(f"Encoded 7-bit codeword: {code}")

    # Introduce a 1-bit error (e.g., flip position 6)
    received = code[:]
    flip_pos = 6  # 1..7
    received[flip_pos-1] ^= 1
    print(f"Received with 1-bit error at pos {flip_pos}: {received}")

    decoded, corrected, err_pos = hamming74_decode(received)
    print(f"Syndrome indicates error position: {err_pos}")
    print(f"Corrected codeword: {corrected}")
    print(f"Decoded data bits: {decoded}")

    print("\n========== 6) Shannon theorems (short explanation) ==========")
    print(shannon_explanations())


def shannon_explanations() -> str:
    return (
        "【夏農信道編碼定理 (Shannon Channel Coding Theorem)】\n"
        "核心結論：對一個固定的「有雜訊信道」，存在一個最大可靠傳輸速率 C（信道容量）。\n"
        "- 如果你的傳輸率 R < C：就能設計某些編碼方式，使得錯誤率可以做到『任意小』（理論上逼近 0）。\n"
        "- 如果 R > C：不管怎麼編碼，錯誤率都不可能趨近 0（一定會有不可避免的錯誤）。\n"
        "這是資訊理論最重要的門檻：C 是『可靠傳輸』的極限。\n\n"
        "【夏農–哈特利定理 (Shannon–Hartley Theorem)】\n"
        "它是針對『加性白高斯雜訊 (AWGN)』通道的一個容量公式：\n"
        "    C = B * log2(1 + S/N)\n"
        "其中：\n"
        "- C：信道容量（bits/s）\n"
        "- B：頻寬（Hz）\n"
        "- S/N：訊號功率 / 雜訊功率（線性比例，不是 dB）\n"
        "直覺：\n"
        "- 頻寬 B 越大，可傳的資訊越多（線性增加）。\n"
        "- SNR 越大，容量增加但呈現 log 增長（報酬遞減）。\n\n"
        "兩者關係：\n"
        "- Shannon 編碼定理是『一般信道』的極限概念（存在容量 C）。\n"
        "- Shannon–Hartley 給出 AWGN 信道下 C 的具體公式。\n"
    )


if __name__ == "__main__":
    main()
