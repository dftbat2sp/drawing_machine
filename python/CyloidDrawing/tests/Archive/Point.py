from dataclasses import dataclass
from tests.Archive.Anchored import Anchored
from numpy import cos, sin


@dataclass
class Point(Anchored):
    x: float
    y: float

    def get_step(self, step: int):
        return self


def rotate_point(point: Point, angle, parent: Point = Point(0, 0)):

    return Point(((point.x * cos(angle)) - (point.y * sin(angle))) + parent.x,
                 ((point.y * cos(angle)) + (point.x * sin(angle))) + parent.y)

class Point2(Anchored):

    def __init__(self, x, y):
        _point = x + y*1j

    def get_x(self):
        return self._point.real

    def get_y(self):
        return self._point.imag
