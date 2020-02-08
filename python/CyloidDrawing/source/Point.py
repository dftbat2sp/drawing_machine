from dataclasses import dataclass
from Anchored import Anchored
from numpy import cos, sin


@dataclass
class Point(Anchored):
    x: float
    y: float

    def get_step(self, step: int):
        return self


def rotate_point(point: Point, angle, origin: Point = Point(0, 0)):

    return Point(((point.x * cos(angle)) - (point.y * sin(angle))) + origin.x,
                 ((point.y * cos(angle)) + (point.x * sin(angle))) + origin.y)
