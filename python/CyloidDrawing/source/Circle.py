from math import radians, tau, pi, sqrt
from cmath import exp, phase
from typing import Type, Union, Tuple
import numpy as np

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

    def __init__(self):
        self.child_point_array: Type[np.ndarray, None] = None
        self.list_calculated = False

    """
    def get_point(self, time_step: float) -> complex:
        raise NotImplementedError
    """
    """
    def get_point_list(self, list_of_base_points: np.ndarray) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError
    """

    def create_point_lists(self, list_of_base_points: np.ndarray) -> None:
        raise NotImplementedError


"""
class PointList:

    def __init__(self, anchorable_object: Type[Anchorable], resolution: float, number_of_rotations: float):
        self.object = anchorable_object
        self.number_of_points = int(number_of_rotations / resolution)
        self.rotation_step_list = [resolution * i for i in range(self.number_of_points)]

    def get_points_list(self):
        return [self.object.get_point(i) for i in self.rotation_step_list]
"""


class BarMateSlide:
    pass


class BarMateFix:

    def __init__(self):
        self.mate_point_array = None

    def get_data_for_circle_intersections(self, time_step: float) -> tuple:
        raise NotImplementedError


class Anchor(Anchorable):

    def __init__(self, complex_number: complex):
        super().__init__()
        self.point = complex_number

    """
    def get_point(self, time_step: float, other_is_bar: bool = False) -> (complex, tuple):
        return self.point
    """

    """
    def get_point_list(self, list_of_base_points: np.ndarray) -> Tuple[np.ndarray, ...]:
        return np.full_like(list_of_base_points, self.point),
    """

    def create_point_lists(self, list_of_base_points):
        self.child_point_array = np.full_like(list_of_base_points, self.point)

    def __add__(self, other):
        return Anchor(self.point + other.point)


# TODO replace one at a time create of points to use numpy.arrays
# see tests.pyg

class Circle(Anchorable, BarMateSlide):
    exp_const = tau * 1j

    def __init__(self, length: float, freq: float, angle: float = 0, deg: bool = False,
                 parent_object: Type[Anchorable] = Anchor(0 + 0j)):
        # circle defined by C * exp( 2*pi*j * freq * resolution_ticker )
        # C = length * exp( starting_angle * j )
        self.rotation_speed: float = freq

        self.child_length: float = length

        if not deg:
            self.starting_angle: float = angle
        else:
            self.starting_angle: float = radians(angle)

        # constant
        self.length_starting_angle: complex = length * exp(self.starting_angle * 1j)

        self.parent: Type[Anchorable] = parent_object

        super().__init__()

    """
    def get_point(self, time_step: float, other_is_bar: bool = False) -> complex:
        return self.parent.get_point(time_step) + \
               self.length_starting_angle * exp(self.exp_const * self.rotation_speed * time_step)
    """

    """
    def get_point_list(self, list_of_base_points: np.ndarray) -> Tuple[np.ndarray, ...]:
        pass
    """

    def create_point_lists(self, list_of_base_points: np.ndarray) -> None:
        if self.parent.points_array is None:
            self.parent.create_point_lists(list_of_base_points)

        self.child_point_array = self.parent.points_array + (
                self.length_starting_angle * exp(self.exp_const * self.rotation_speed * list_of_base_points))

    def __str__(self):
        return f'length: {self.child_length}, freq: {self.rotation_speed}, angle: {self.starting_angle}'


class Bar(Anchorable, BarMateFix):

    def __init__(self, parent_object, child_length_from_parent, mate_object, mate_length_from_parent):
        super().__init__()
        self.parent: Type[Anchorable] = parent_object
        self.mate: Type[Anchorable, BarMateSlide, BarMateFix] = mate_object
        self.child_length = child_length_from_parent
        self.mate_length = mate_length_from_parent

    """
    def get_point(self, time_step: float):
        parent = self.parent.get_point(time_step)

        if isinstance(self.mate, BarMateFix):
            mate_parent, mate_length = self.mate.get_data_for_circle_intersections(time_step)

            mate = get_circles_intersections(parent.real, parent.imag, self.mate_length,
                                             mate_parent.real, mate_parent.imag, mate_length)
            
        elif isinstance(self.mate, BarMateSlide):
            mate = self.mate.get_point(time_step)

        mate_parent_diff = mate - parent
        angle = phase(mate_parent_diff)

        return parent + (self.child_length * exp(angle * 1j))
    """

    def create_point_lists(self, list_of_base_points: np.ndarray) -> None:
        if self.parent.child_point_array is None:
            self.parent.create_point_lists(list_of_base_points)

        if self.mate.child_point_array is None:
            self.mate.create_point_lists(list_of_base_points)

        #  mate is a Mate Fix
        if isinstance(self.mate, BarMateFix):
            pass
        elif isinstance(self.mate, BarMateSlide):
            pass

    def get_data_for_circle_intersections(self, time_step: float) -> tuple:
        pass
        # parent_point = self.parent.get_point(time_step)
        # return parent_point, self.mate_length

    def get_circle_intersection_with_mate(self):

        d: np.ndarray = np.sqrt(np.power((self.child_point_array.real - self.mate.child_point_array.real), 2) +
                    np.sqrt(np.power((self.child_point_array.imag - self.mate.child_point_array.imag), 2)))

        if (d > (self.mate_length + self.mate.mate_length)).any():
            raise IntersectionError(f'Non-intersecting, non-concentric circles not contained within each other.')

        elif (d < abs(self.mate_length - self.mate.mate_length)).any():
            raise IntersectionError(f'Non-intersecting circles. One contained in the other.')

        elif (d == 0).any():
            raise IntersectionError(f'Concentric circles.')
        else:



"""
class Draw:

    def __init__(self, draw_obj, resolution, num_of_rotations, *supporting_objs):

        self.draw_obj = draw_obj
        self.supporting_obj = supporting_objs

        self.resolution = resolution
        self.num_of_rotations = num_of_rotations

"""

anchor1 = Anchor(3 + 0j)
# anchor1 = Anchor(0)
anchor2 = Anchor(-3 - 0j)

c1 = Circle(1.5, .7, parent_object=anchor1)
c2 = Circle(0.8, .5, parent_object=anchor2)
bar2 = Bar(c1, 8, None, 6)
b1 = Bar(c2, 9, bar2, 7)
bar2.mate = b1
# b1 = Bar(anchor1, c1, 1.5)

res = 0.001
rot = 10.0

n1 = [b1.get_point(i) for i in range(20000000)]

print('Done')
import time

time.sleep(20)
# Draw(main_obj, *everything_else)

# circles -> draw child point
# bars -> draw parent to child, dot on mate


"""






# Animation
b2 = PointList(b1, res, rot)
b2_points = b2.get_points_list()
x = [i.real for i in b2_points]
y = [i.imag for i in b2_points]
z = [x,y]
min_x = min(x)
max_x = max(x)
width_x = max_x - min_x

min_y = min(y)
max_y = max(y)
width_y = max_y - min_y

extra_width = 7

fig = pyplot.figure(figsize=(width_x * 3, width_y * 3))
ax = pyplot.axes(xlim=(min_x - extra_width, max_x + extra_width), ylim=(min_y - extra_width, max_y + extra_width))
ax.set_aspect('equal')

line, = ax.plot([], [])
mark, = ax.plot([], [], marker='o', markersize=3, color='r')

b3 = PointList(c1, res, rot)
b4 = PointList(c2, res, rot)

c1_points = b3.get_points_list()
c2_points = b4.get_points_list()

c1_x = [i.real for i in c1_points]
c1_y = [i.imag for i in c1_points]

c2_x = [i.real for i in c2_points]
c2_y = [i.imag for i in c2_points]

c1_line, = ax.plot([], [])
c1_mark, = ax.plot([], [], marker='o', markersize=3, color='b')
c2_line, = ax.plot([], [])
c2_mark, = ax.plot([], [], marker='o', markersize=3, color='g')

myline, = ax.plot([], [])


# pen_c1.plot()
# pen_c2.plot()

def init():  # only required for blitting to give a clean slate.
    # line.set_ydata([np.nan] * len(x))
    return pyplot.axes().plot([], [])
    # line.set_data([], [])
    # mark.set_data([], [])
    # c1_line.set_data([], [])
    # c1_mark.set_data([], [])
    # c2_line.set_data([], [])
    # c2_mark.set_data([], [])
    #
    # myline.set_data([], [])
    # return line, mark, c1_line, c1_mark, c2_line, c2_mark, myline
    # return line,
    # return matplotlib.lines.Line2D(np.cos(x[:1]), np.sin(x[:1])),


def get_frames():
    for i in range(len(x)):
        point = i * 20
        if point < len(x):
            yield point


def animate(i):
    # y = np.arange(0 + (i/np.pi), (2 * np.pi) - 0.5 + (i/np.pi), 0.01)
    # line.set_ydata(np.sin(y))  # update the data.
    # line.set_xdata(np.cos(y))
    # line2, = ax.plot(np.cos(x[:i]), np.sin(x[:i]))

    # i = i * 50



    line.set_data(z[0][:i], z[1][:i])
    mark.set_data(z[0][i], z[1][i])

    # c1_line.set_data(c1_x[:i], c1_y[:i])
    # c1_mark.set_data(c1_x[i], c1_y[i])
    # c2_line.set_data(c2_x[:i], c2_y[:i])
    # c2_mark.set_data(c2_x[i], c2_y[i])
    #
    # myline.set_data([c2_x[i], x[i]], [c2_y[i], y[i]])



    return line, mark, #c1_line, c1_mark, c2_line, c2_mark, myline


ani = mpl_animation.FuncAnimation(fig, animate, init_func=init, interval=1, blit=True, save_count=1, frames=get_frames,
                                  repeat=False,)

pyplot.show()
"""

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

"""
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
"""
