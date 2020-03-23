from math import radians, tau, pi
from cmath import exp, phase
from typing import Type

import matplotlib.pyplot as pyplot


class Anchorable:

    def get_point(self, time_step: float) -> complex:
        return NotImplementedError


class Anchor(Anchorable):

    def __init__(self, complex_number: complex):
        self.point = complex_number

    def get_point(self, time_step: float) -> complex:
        return self.point


class Circle(Anchorable):
    exp_const = tau * 1j

    def __init__(self, length: float, freq: float, angle: complex = 0 + 0j, deg: bool = False,
                parent_object: Type[Anchorable] = Anchor(0 + 0j)):
        # circle defined by C * exp( 2*pi*j * freq * resolution_ticker )
        # C = length * exp( starting_angle * j )
        self.rotation_speed: float = freq

        self.length: float = length

        if not deg:
            self.starting_angle: float = angle
        else:
            self.starting_angle: float = radians(angle)

        # constant
        self.length_starting_angle: complex = length * exp(self.starting_angle * 1j)

        self.parent: Type[Anchorable] = parent_object

        super().__init__()

    def get_point(self, time_step: float) -> complex:
        return self.parent.get_point(time_step) + \
               self.length_starting_angle * exp(self.exp_const * self.rotation_speed * time_step)

    def __str__(self):
        return f'length: {self.length}, freq: {self.rotation_speed}, angle: {self.starting_angle}'

class Bar(Anchorable):

    def __init__(self, parent_object, mate_object, child_length_from_parent):
        self.parent_object = parent_object
        self.mate_object = mate_object
        self.length = child_length_from_parent

    def get_point(self, time_step: float) -> complex:
        parent = self.parent_object.get_point(time_step)
        mate = self.mate_object.get_point(time_step)

        mate_parent_diff = mate - parent
        angle = phase(mate_parent_diff)

        return self.length * exp(angle * 1j)


class Drawer:

    def __init__(self, draw_object, resolution: float, number_of_rotations: float):
        self.object = draw_object
        self.number_points = int(number_of_rotations / resolution)
        self.point_step_list = [resolution * i for i in range(self.number_points)]

        self.fig, self.ax = pyplot.subplots()
        self.ax.set_aspect('equal')

    def plot(self):
        points_list = [self.object.get_point(i) for i in self.point_step_list]
        x_list = [i.real for i in points_list]
        y_list = [i.imag for i in points_list]

        self.ax.plot(x_list, y_list)

        pyplot.show()



anchor = Anchor(0)

c1 = Circle(2.0, 1.0)
c2 = Circle(1.0, pi, angle=pi, parent_object=c1)

pen = Drawer(c2, 0.01, 5)
pen.plot()
