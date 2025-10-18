# geometry2d.py
# 2D analytic geometry utilities — points, lines, circles, triangles
from __future__ import annotations
from dataclasses import dataclass
from math import hypot, sqrt, cos, sin, isclose, atan2
from typing import Optional, Tuple, List

EPS = 1e-9
def close(a: float, b: float, tol: float = EPS) -> bool:
    return isclose(a, b, rel_tol=0.0, abs_tol=tol)

# -------------------- Core primitives --------------------

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    # vector ops
    def __add__(self, other: "Point") -> "Point": return Point(self.x + other.x, self.y + other.y)
    def __sub__(self, other: "Point") -> "Point": return Point(self.x - other.x, self.y - other.y)
    def __mul__(self, k: float) -> "Point": return Point(self.x * k, self.y * k)
    __rmul__ = __mul__

    def dot(self, other: "Point") -> float: return self.x * other.x + self.y * other.y
    def cross(self, other: "Point") -> float: return self.x * other.y - self.y * other.x
    def norm(self) -> float: return hypot(self.x, self.y)
    def dist(self, other: "Point") -> float: return (self - other).norm()

    # transforms
    def translate(self, dx: float, dy: float) -> "Point": return Point(self.x + dx, self.y + dy)
    def scale(self, sx: float, sy: Optional[float] = None, origin: "Point" = None) -> "Point":
        if sy is None: sy = sx
        if origin is None: origin = Point(0.0, 0.0)
        v = self - origin
        return Point(origin.x + v.x * sx, origin.y + v.y * sy)
    def rotate(self, theta: float, origin: "Point" = None) -> "Point":
        if origin is None: origin = Point(0.0, 0.0)
        v = self - origin
        c, s = cos(theta), sin(theta)
        return Point(origin.x + c * v.x - s * v.y, origin.y + s * v.x + c * v.y)


@dataclass(frozen=True)
class Line:
    """
    Infinite line in ax + by + c = 0 form (normalized for stability).
    """
    a: float
    b: float
    c: float

    @staticmethod
    def through(p: Point, q: Point) -> "Line":
        # (y - y1) = m (x - x1)  -> ax+by+c=0 with a = y1 - y2, b = x2 - x1
        a = p.y - q.y
        b = q.x - p.x
        c = -(a * p.x + b * p.y)
        # normalize
        norm = hypot(a, b)
        if norm < EPS: raise ValueError("Points are identical; no unique line.")
        a, b, c = a / norm, b / norm, c / norm
        # make a non-negative for determinism
        if a < 0 or (close(a, 0) and b < 0):
            a, b, c = -a, -b, -c
        return Line(a, b, c)

    @staticmethod
    def from_point_dir(p: Point, d: Point) -> "Line":
        if d.norm() < EPS: raise ValueError("Direction must be nonzero.")
        q = p + d
        return Line.through(p, q)

    def direction(self) -> Point:
        # perpendicular to normal (a, b)
        return Point(self.b, -self.a)

    def project(self, p: Point) -> Point:
        # foot of perpendicular from point p to the line
        # formula: p - n * (ax0 + by0 + c) where n = (a,b) / ||(a,b)||^2, but a,b are normalized => denom = 1
        t = self.a * p.x + self.b * p.y + self.c
        return Point(p.x - self.a * t, p.y - self.b * t)

    def is_parallel(self, other: "Line") -> bool:
        return close(self.a * other.b - self.b * other.a, 0)

    def intersection(self, other: "Line") -> Optional[Point]:
        det = self.a * other.b - self.b * other.a
        if close(det, 0):  # parallel or coincident
            return None
        x = (self.b * other.c - self.c * other.b) / det
        y = (self.c * other.a - self.a * other.c) / det
        return Point(x, y)

    def distance_to_point(self, p: Point) -> float:
        return abs(self.a * p.x + self.b * p.y + self.c)  # because (a,b) is unit-length


@dataclass(frozen=True)
class Circle:
    center: Point
    r: float

    def contains(self, p: Point, tol: float = EPS) -> bool:
        return close(self.center.dist(p), self.r, tol)

# -------------------- Intersections --------------------

def intersect_line_line(L1: Line, L2: Line) -> Optional[Point]:
    return L1.intersection(L2)

def intersect_line_circle(L: Line, C: Circle) -> List[Point]:
    """
    Return 0, 1, or 2 intersection points of a line and a circle.
    """
    # Foot of perpendicular from center to line
    H = L.project(C.center)
    d = C.center.dist(H)
    if d > C.r + EPS:
        return []
    if close(d, C.r):
        return [H]
    # Along the line direction
    offset = sqrt(max(C.r*C.r - d*d, 0.0))
    dir_vec = L.direction()  # already unit length because Line is normalized
    # ensure it's unit: ||(b,-a)|| = 1
    p1 = Point(H.x + dir_vec.x * offset, H.y + dir_vec.y * offset)
    p2 = Point(H.x - dir_vec.x * offset, H.y - dir_vec.y * offset)
    return [p1, p2]

def intersect_circle_circle(C1: Circle, C2: Circle) -> List[Point]:
    """
    Return 0, 1, or 2 intersection points of two circles.
    """
    d = C1.center.dist(C2.center)
    r1, r2 = C1.r, C2.r
    if d > r1 + r2 + EPS or d < abs(r1 - r2) - EPS or d < EPS:
        return []  # separate, contained, or same center (degenerate)
    # distance from C1.center to line of centers' intersection points
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a**2
    if h_sq < -EPS:
        return []
    h = sqrt(max(h_sq, 0.0))
    # base point P along the center line
    ex = (C2.center - C1.center) * (1 / d)
    P = C1.center + ex * a
    # perpendicular offset
    ey = Point(-ex.y, ex.x)
    if close(h, 0):
        return [P]  # tangent
    return [P + ey * h, P - ey * h]

# -------------------- Perpendicular from point to line --------------------

def perpendicular_from_point_to_line(P: Point, L: Line) -> Tuple[Line, Point]:
    """Return the perpendicular line and the foot (projection) point."""
    H = L.project(P)
    # perpendicular line through P has direction normal to L: (a, b)
    perp_dir = Point(L.a, L.b)  # normal vector
    perp_line = Line.from_point_dir(P, perp_dir)
    return perp_line, H

# -------------------- Pythagoras verification --------------------

def verify_pythagoras(L: Line, P_outside: Point, A_on_line: Point) -> bool:
    """
    Using the right triangle A–H–P where H is the foot of the perpendicular from P to line L:
    check |AP|^2 == |AH|^2 + |HP|^2.
    """
    _, H = perpendicular_from_point_to_line(P_outside, L)
    AP2 = A_on_line.dist(P_outside) ** 2
    AH2 = A_on_line.dist(H) ** 2
    HP2 = H.dist(P_outside) ** 2
    return close(AP2, AH2 + HP2, tol=1e-7)

# -------------------- Triangle object --------------------

@dataclass(frozen=True)
class Triangle:
    A: Point
    B: Point
    C: Point

    def sides(self) -> Tuple[float, float, float]:
        return (self.B.dist(self.C), self.C.dist(self.A), self.A.dist(self.B))  # a=|BC|, b=|CA|, c=|AB|

    def perimeter(self) -> float:
        a, b, c = self.sides()
        return a + b + c

    def area(self) -> float:
        # Heron's formula
        a, b, c = self.sides()
        s = (a + b + c) / 2
        return sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))

    def is_right(self, tol: float = 1e-7) -> bool:
        a, b, c = sorted(self.sides())
        return close(c*c, a*a + b*b, tol)

    # transforms return new Triangle
    def translate(self, dx: float, dy: float) -> "Triangle":
        return Triangle(self.A.translate(dx, dy), self.B.translate(dx, dy), self.C.translate(dx, dy))
    def scale(self, s: float, origin: Point = Point(0.0, 0.0)) -> "Triangle":
        return Triangle(self.A.scale(s, s, origin), self.B.scale(s, s, origin), self.C.scale(s, s, origin))
    def rotate(self, theta: float, origin: Point = Point(0.0, 0.0)) -> "Triangle":
        return Triangle(self.A.rotate(theta, origin), self.B.rotate(theta, origin), self.C.rotate(theta, origin))

# -------------------- Demonstration --------------------
if __name__ == "__main__":
    # 1) Define point, line, circle
    P = Point(3, 4)
    Q = Point(-2, 1)
    L = Line.through(P, Q)            # line PQ
    C = Circle(center=Point(1, 1), r=3)

    # 2) Intersections
    # line-line
    L2 = Line.through(Point(0, 0), Point(1, 2))
    print("Line-Line intersection:", intersect_line_line(L, L2))

    # line-circle
    print("Line-Circle intersections:", intersect_line_circle(L2, C))

    # circle-circle
    C2 = Circle(Point(3, 1), 2.5)
    print("Circle-Circle intersections:", intersect_circle_circle(C, C2))

    # 3) Perpendicular from point to a line
    P_out = Point(2, 5)
    perp, H = perpendicular_from_point_to_line(P_out, L2)
    print("Perpendicular line (a,b,c):", perp)
    print("Foot of perpendicular H:", H)

    # 4) Verify Pythagoras with A on L2
    A = Point(0, 0)  # A is on L2
    print("Pythagoras holds?", verify_pythagoras(L2, P_out, A))

    # 5) Triangle object and 6) transforms
    T = Triangle(A=Point(0, 0), B=Point(3, 0), C=Point(0, 4))
    print("Triangle area:", T.area(), "perimeter:", T.perimeter(), "is right?", T.is_right())
    print("Triangle rotated 90° around origin:", T.rotate(3.141592653589793/2))
