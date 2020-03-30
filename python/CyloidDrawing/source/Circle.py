from math import radians, tau, pi, sqrt
from cmath import exp, phase
from typing import Type, Union

import matplotlib.pyplot as pyplot
import matplotlib.animation as mpl_animation


class IntersectionError(Exception):
    pass


def get_circles_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1:
        raise IntersectionError(f'Non-intersecting, non-concentric circles not contained within each other.')
    # One circle within other
    if d < abs(r0 - r1):
        raise IntersectionError(f'Non-intersecting circles. One contained in the other.')
    # coincident circles
    if d == 0:
        raise IntersectionError(f'Concentric circles.')
    else:
        # to ensure a bias to top and right
        if x0 < x1:
            x0, x1 = x1, x0
            r0, r1 = r1, r0
        if y0 > y1:
            y0, y1 = y1, y0
            r0, r1 = r1, r0

        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        return x3 + y3 * 1j


class Anchorable:

    def get_point(self, time_step: float) -> complex:
        raise NotImplementedError


class PointList:

    def __init__(self, anchorable_object: Type[Anchorable], resolution: float, number_of_rotations: float):
        self.object = anchorable_object
        self.number_of_points = int(number_of_rotations / resolution)
        self.rotation_step_list = [resolution * i for i in range(self.number_of_points)]

    def get_points_list(self):
        return [self.object.get_point(i) for i in self.rotation_step_list]


class Anchor(Anchorable):

    def __init__(self, complex_number: complex):
        self.point = complex_number

    def get_point(self, time_step: float, other_is_bar: bool = False) -> (complex, tuple):
        return self.point


class Circle(Anchorable):
    exp_const = tau * 1j

    def __init__(self, length: float, freq: float, angle: float = 0, deg: bool = False,
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

    def get_point(self, time_step: float, other_is_bar: bool = False) -> complex:
        return self.parent.get_point(time_step) + \
               self.length_starting_angle * exp(self.exp_const * self.rotation_speed * time_step)

    def __str__(self):
        return f'length: {self.length}, freq: {self.rotation_speed}, angle: {self.starting_angle}'


class Bar(Anchorable):

    def __init__(self, parent_object, mate_object, child_length_from_parent):
        self.parent_object = parent_object
        self.mate_object = mate_object
        self.length = child_length_from_parent

    def get_point(self, time_step: float, other_is_bar: bool = False):
        parent = self.parent_object.get_point(time_step)
        if other_is_bar:
            return parent, self.length,
        else:
            mate = self.mate_object.get_point(time_step, True)

        if isinstance(mate, tuple):
            mate_parent = mate[0]
            mate_length = mate[1]

            mate_point = get_circles_intersections(parent.real, parent.imag, self.length,
                                                   mate_parent.real, mate_parent.imag, mate_length)
        elif isinstance(mate, complex):
            mate_point = mate

        mate_parent_diff = mate_point - parent
        angle = phase(mate_parent_diff)

        return parent + (self.length * exp(angle * 1j))


class Drawer(PointList):

    def __init__(self, draw_object, resolution: float, number_of_rotations: float):
        super().__init__(draw_object, resolution, number_of_rotations)
        # self.object = draw_object
        # self.number_of_points = int(number_of_rotations / resolution)
        # self.rotation_step_list = [resolution * i for i in range(self.number_of_points)]

        self.ax = pyplot.subplot()
        self.ax.set_aspect('equal')

    def plot(self):
        # points_list = [self.object.get_point(i) for i in self.rotation_step_list]
        points_list = self.get_points_list()
        x_list = [i.real for i in points_list]
        y_list = [i.imag for i in points_list]

        # self.ax.plot(x_list, y_list)
        pyplot.plot(x_list, y_list)


anchor1 = Anchor(5 + 5j)
# anchor1 = Anchor(0)
anchor2 = Anchor(-2 - 2j)

c1 = Circle(pi/2, pi / 4, parent_object=anchor1)
c2 = Circle(1.0, 1.0, parent_object=anchor2)
b1 = Bar(c2, c1, 3)
# b1 = Bar(anchor1, c1, 1.5)

res = 0.001
rot = 20.0

# Animation
b2 = PointList(b1, res, rot)
b2_points = b2.get_points_list()
x = [i.real for i in b2_points]
y = [i.imag for i in b2_points]

fig = pyplot.figure()
ax = pyplot.axes(xlim=(-5, 5), ylim=(-5, 5))
ax.set_aspect('equal')

line, = ax.plot([], [])


def init():  # only required for blitting to give a clean slate.
    # line.set_ydata([np.nan] * len(x))
    line.set_data([], [])
    return line,
    # return matplotlib.lines.Line2D(np.cos(x[:1]), np.sin(x[:1])),


def animate(i):
    # y = np.arange(0 + (i/np.pi), (2 * np.pi) - 0.5 + (i/np.pi), 0.01)
    # line.set_ydata(np.sin(y))  # update the data.
    # line.set_xdata(np.cos(y))
    # line2, = ax.plot(np.cos(x[:i]), np.sin(x[:i]))
    i = i * 4
    line.set_data(x[:i], y[:i])
    return line,


ani = mpl_animation.FuncAnimation(fig, animate, init_func=init, interval=1, blit=True, save_count=50)

pen_c1 = Drawer(c1, res, rot)
pen_c2 = Drawer(c2, res, rot)

pen_c1.plot()
pen_c2.plot()

pyplot.show()
"""
# Drawers
pen_b1 = Drawer(b1, res, rotations)
pen_c1 = Drawer(c1, res, rotations)
pen_c2 = Drawer(c2, res, rotations)

pen_b1.plot()
pen_c1.plot()
pen_c2.plot()

pyplot.show()
"""
