from SimpleVector import SimpleVector
from sympy import Rational, rad
from typing import Union
from Angle import Angle

class StepCounterError(Exception):
    pass

class Rotor:

    def __init__(self, length, rpm, starting_angle, resolution = 0.1, deg: bool = False, parent_object = SimpleVector(0, 0)):

        #unless otherwise set, vector starts from the origin

        #universal
        self.parent_object = parent_object
        self.parent_vector: SimpleVector = parent_object.get_step(0)
        self.length = length
        self.rpm = rpm
        self.rpm_step_ratio = Rational(rpm, rpm)
        self.starting_angle = starting_angle
        if deg:
            self.starting_angle = rad(starting_angle)

        #universal
        self.resolution = resolution

        # universal
        self.child_vector: SimpleVector = SimpleVector(self.length, 0)
        self.child_vector.rotate_vector(self.starting_angle)
        self.base_child_vector = self.child_vector

        # universal
        self.current_step: int = 0

    # def get_step(self, step_count: int):
    #     if step_count == self.current_step:
    #         return self.parent_vector + self.child_vector
    #     if step_count - 1 == self.current_step:
    #         self.current_step += 1
    #         self.parent_vector = self.parent_object.get_step(step_count)
    #
    #         # rotor
    #         # (resolution * rpm_step_ratio) + starting_angle
    #         self.child_vector = self.base_child_vector
    #         self.child_vector.rotate_vector(step_count * self.rpm_step_ratio * self.resolution)
    #
    #         return self.child_vector
    #     else:
    #         raise StepCounterError

