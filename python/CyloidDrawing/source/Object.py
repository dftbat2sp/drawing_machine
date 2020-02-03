from Point import Point
from Anchored import Anchored
from Angle import Angle
import numpy as np

class Object(Anchored):
    def __init__(self, pa):
        super(Object, self).__init__()
        self.parent_anchor: Point = pa
        self.child_anchor: Point = Point(0, 0)
        # change flag to see if item has been previously calculated this "round"
        self.step_counter: int = 0

    def get_child_anchor(self):
        # get child anchor of parent
        # then do calculation base
        raise NotImplementedError

    @staticmethod
    def new_point_from_angle_and_length(starting_point: Point, angle: Angle, length):
        change_x = length * np.cos(angle.magnitude)
        change_y = length * np.sin(angle.magnitude)
        new_x = starting_point.x + change_x
        new_y = starting_point.y + change_y
        return Point(new_x, new_y)