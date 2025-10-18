# finite_field.py
# A minimal finite field F_p (p prime) with verification of axioms.

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import math
import random

# ---------- number helpers ----------

def egcd(a: int, b: int) -> Tuple[int,int,int]:
    if b == 0:
        return a, 1, 0
    g, x, y = egcd(b, a % b)
    return g, y, x - (a // b) * y

def modinv(a: int, p: int) -> int:
    a %= p
    g, x, _ = egcd(a, p)
    if g != 1:
        raise ZeroDivisionError(f"{a} has no inverse mod {p}")
    return x % p

def is_prime(p: int) -> bool:
    if p < 2: return False
    small = [2,3,5,7,11,13,17,19,23,29]
    for q in small:
        if p == q: return True
        if p % q == 0: return False
    # Miller–Rabin (deterministic for 32-bit and enough for coursework)
    d, s = p - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in [2, 7, 61]:  # good bases for 32-bit, quick and fine for class
        if a % p == 0:
            continue
        x = pow(a, d, p)
        if x == 1 or x == p - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % p
            if x == p - 1:
                break
        else:
            return False
    return True

# ---------- Finite field core ----------

class FiniteField:
    """Finite field F_p with p prime elements {0,1,...,p-1}."""
    def __init__(self, p: int):
        if not is_prime(p):
            raise ValueError("p must be prime for F_p")
        self.p = p

    def element(self, x: int | 'FF') -> 'FF':
        return x if isinstance(x, FF) and x.field is self else FF(int(x) % self.p, self)

    def zero(self) -> 'FF': return FF(0, self)
    def one(self)  -> 'FF': return FF(1, self)

    # convenience iteration
    def elements(self) -> List['FF']:
        return [FF(i, self) for i in range(self.p)]

    def nonzero(self) -> List['FF']:
        return [FF(i, self) for i in range(1, self.p)]

@dataclass(frozen=True)
class FF:
    """An element of F_p with full operator overloading."""
    value: int
    field: FiniteField

    # internal normalize
    def _n(self, a: int) -> int: return a % self.field.p

    # coercion for rhs
    def _coerce(self, other) -> 'FF':
        if isinstance(other, FF):
            if other.field is not self.field:
                raise TypeError("Cannot mix elements from different fields")
            return other
        return FF(int(other) % self.field.p, self.field)

    # pretty printing / int
    def __int__(self): return self.value % self.field.p
    def __repr__(self): return f"FF({int(self)}, p={self.field.p})"

    # equality
    def __eq__(self, other): return int(self) == int(self._coerce(other))

    # group operations
    def __neg__(self): return FF((-int(self)) % self.field.p, self.field)
    def __add__(self, other): other = self._coerce(other); return FF((int(self)+int(other))%self.field.p, self.field)
    def __sub__(self, other): other = self._coerce(other); return FF((int(self)-int(other))%self.field.p, self.field)
    def __mul__(self, other): other = self._coerce(other); return FF((int(self)*int(other))%self.field.p, self.field)

    # power and division
    def __pow__(self, k: int): return FF(pow(int(self), k, self.field.p), self.field)
    def inv(self): 
        if int(self) == 0: raise ZeroDivisionError("0 has no inverse")
        return FF(modinv(int(self), self.field.p), self.field)
    def __truediv__(self, other): other = self._coerce(other); return self * other.inv()

# ---------- Axiom checks (group_axioms.py style) ----------

def is_group_under_add(field: FiniteField) -> bool:
    E = field.elements()
    zero = field.zero()
    # closure, associativity, identity, inverse, commutativity
    p = field.p
    # associativity & commutativity: explicit finite check
    for a in E:
        for b in E:
            # closure
            if not isinstance(a + b, FF): return False
            # commutative
            if (a + b) != (b + a): return False
            for c in E:
                if (a + (b + c)) != ((a + b) + c): return False
    # identity & inverse
    for a in E:
        if (a + zero) != a or (zero + a) != a: return False
        # inverse exists
        inv = None
        for b in E:
            if a + b == zero:
                inv = b; break
        if inv is None: return False
    return True

def is_group_under_mul_nonzero(field: FiniteField) -> bool:
    E = field.nonzero()
    one = field.one()
    # closure, associativity, identity, inverse, commutativity (F_p is commutative)
    for a in E:
        for b in E:
            if int(a*b) == 0:  # closure in nonzero set
                return False
            if (a * b) != (b * a): return False
            for c in E:
                if (a * (b * c)) != ((a * b) * c): return False
    for a in E:
        if (a * one) != a or (one * a) != a: return False
        try:
            _ = a.inv()
        except ZeroDivisionError:
            return False
    return True

def check_distributivity(field: FiniteField) -> bool:
    E = field.elements()
    for a in E:
        for b in E:
            for c in E:
                if a * (b + c) != (a * b + a * c): return False
                if (a + b) * c != (a * c + b * c): return False
    return True

# ---------- Demonstration & quick tests ----------

if __name__ == "__main__":
    F7 = FiniteField(7)

    print("Additive group OK? ", is_group_under_add(F7))
    print("Multiplicative group (nonzero) OK? ", is_group_under_mul_nonzero(F7))
    print("Distributivity OK? ", check_distributivity(F7))

    # Using the field object like numbers
    a, b, c = F7.element(3), F7.element(5), F7.element(2)
    print("\nArithmetic demo in F_7:")
    print("a =", a, "b =", b, "c =", c)
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a * b =", a * b)
    print("b / c =", b / c)
    print("a**10 =", a**10)
    # Small verification: (a+b)*c == a*c + b*c
    print("(a+b)*c == a*c + b*c ?", (a+b)*c == a*c + b*c)
