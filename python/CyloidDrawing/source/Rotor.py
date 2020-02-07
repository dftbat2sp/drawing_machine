from abc import ABC

from sympy import Rational, rad, Point2D
from typing import Union


class StepCounterError(Exception):
    pass


class Point2DExtended(Point2D, ABC):

    def __init__(self, x, y):
        super().__init__(x, y)

    def get_step(self, step_count: int):
        return self


class Rotor:

    def __init__(self, length, rpm, angle, resolution=0.1, deg: bool = False, parent_object=Point2DExtended(0, 0)):
        # unless otherwise set, vector starts from the origin

        # universal
        self.parent_object = parent_object
        self.length = length
        self.rpm = rpm
        self.rpm_step_ratio = Rational(rpm, rpm)
        self.starting_angle = angle
        if deg:
            self.starting_angle = rad(angle)

        # universal
        self.resolution = resolution

        self.starting_point = Point2DExtended(length, 0).rotate(self.starting_angle, self.parent_object.get_step(0))

        # universal
        # self.child_vector: SimpleVector = SimpleVector(self.length, 0)
        # self.child_vector.rotate_vector(self.starting_angle)
        # self.base_child_vector = self.child_vector

        # universal
        # self.current_step: int = 0

    def get_step(self, step_count: int):
        return self.starting_point.rotate(step_count * self.rpm_step_ratio * self.resolution,
                                          self.parent_object.get_step(step_count))

        # parent_point = self.parent_object.get_step(step_count)
        #
        # # rotor
        # self.child_vector.rotate_vector(step_count * self.rpm_step_ratio * self.resolution)
        #
        # return self.child_vector
