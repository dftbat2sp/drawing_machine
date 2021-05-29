# import cmath
import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Type, Union
import random

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


class IntersectionError(Exception):
    pass


class SpiralTypeError(Exception):
    pass


"""
def get_circles_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1:
        raise IntersectionError(
            f'Non-intersecting, non-concentric circles not contained within each other.')
    # One circle within other
    if d < abs(r0 - r1):
        raise IntersectionError(
            f'Non-intersecting circles. One contained in the other.')
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
        h = math.sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        return x3 + y3 * 1j
"""

color_hex = ['#264653',
             '#2a9d8f',
             '#e76f51',
             '#e76f51',
             '#e63946',
             '#06d6a0',
             '#118ab2',
             '#f4acb7',
             '#8338ec',
             '#9a031e',
             '#fb5607',
             '#22223b',
             '#3a0ca3',
             '#3a0ca3']


def remove_duplicates_from_list(my_list):
    return list(dict.fromkeys(my_list))


@dataclass
class RotationResolution:
    step_size: float = 0.001
    rotations: float = 50.0
    balanced_around_zero: bool = False

    def __post_init__(self):
        self.num_of_points: int = int(self.rotations / self.step_size)
        self.num_of_rotation_in_radians: float = self.rotations * math.tau
        if self.balanced_around_zero:
            half_rotation_in_radian = self.num_of_rotation_in_radians / 2
            self.radial_steps_list: np.ndarray = np.linspace(
                -half_rotation_in_radian, half_rotation_in_radian, self.num_of_points)
        else:
            self.radial_steps_list: np.ndarray = np.linspace(
                0, self.num_of_rotation_in_radians, self.num_of_points)


class Anchorable:

    def __init__(self):
        self.point_array: Type[Union[np.ndarray, None]] = None
        # self.list_calculated = False

    def create_point_lists(self, base_points_list: RotationResolution) -> None:
        raise NotImplementedError

    def update_drawing_objects(self, frame) -> None:
        raise NotImplementedError

    def get_main_drawing_objects(self) -> List:
        raise NotImplementedError

    def get_secondary_drawing_objects(self) -> List:
        raise NotImplementedError

    def get_min_max_values(self, buffer: float = 0, point_array_only: bool = False, point_array_starting_point: int = 0) -> Tuple:
        """
        returns the min and max values
        Returns
        -------
        Tuple with 4 values:
            Minimum X Value
            Maximum X Value
            Minimum Y Value
            Maximum Y Value
        """
        x_min = min(
            self.point_array[point_array_starting_point:].real) - buffer
        x_max = max(
            self.point_array[point_array_starting_point:].real) + buffer

        y_min = min(
            self.point_array[point_array_starting_point:].imag) - buffer
        y_max = max(
            self.point_array[point_array_starting_point:].imag) + buffer

        return x_min, x_max, y_min, y_max

    def get_min_max_values_normalized_to_origin(self, buffer: float = 0) -> Tuple:
        """
        Get the min/max xy values of object if parent is moved to zero
        Returns
        -------
        Tuple with 4 values:
            Minimum X Value if parent was at origin
            Maximum X Value if parent was at origin
            Minimum Y Value if parent was at origin
            Maximum Y Value if parent was at origin
        """
        raise NotImplementedError

    def get_parent_tree(self) -> List:
        # get list of parent heirarchy
        parent_tree = self.parent.get_parent_tree()
        # add self to list
        parent_tree.append(self)
        # filter list for duplicates
        parent_tree = remove_duplicates_from_list(parent_tree)

        return parent_tree

    def animate(self, resolution_obj: RotationResolution, speed: int = 1, point_array_starting_point: int = 0):

        # get full parent tree of drawer (list of obj)
        obj_needed_for_drawing = self.get_parent_tree()

        # get our own points list
        self.create_point_lists(resolution_obj)

        # calculate points needed to draw all objs
        # for obj in obj_needed_for_drawing:
        # obj.create_point_lists(resolution_obj)

        # get artists for drawing on mpl figure
        artist_list = []

        for obj in obj_needed_for_drawing:
            artist_list.extend(obj.get_main_drawing_objects())

        # add figure and subplots to makes axes
        fig = plt.figure(figsize=[6, 6])
        fig_static = plt.figure(figsize=[6, 6])

        ax = fig.add_subplot(111, frameon=False)
        ax_static = fig_static.add_subplot(111, frameon=False)

        fig.tight_layout()
        fig_static.tight_layout()

        plt.axis('off')

        # get figure limits from drawing objs
        xmin, xmax, ymin, ymax = self.get_min_max_values(
            point_array_starting_point=point_array_starting_point)

        for obj in obj_needed_for_drawing:
            obj_xmin, obj_xmax, obj_ymin, obj_ymax = obj.get_min_max_values(
                point_array_starting_point=point_array_starting_point)
            xmin = min(xmin, obj_xmin)
            xmax = max(xmax, obj_xmax)
            ymin = min(ymin, obj_ymin)
            ymax = max(ymax, obj_ymax)

        limits_buffer = 5
        xmin = xmin - limits_buffer
        xmax = xmax + limits_buffer
        ymin = ymin - limits_buffer
        ymax = ymax + limits_buffer

        # prevent the limits from exceeding a vallue
        # helps with + and - infinity limitss
        max_limit = 1000
        xmin = max(-max_limit, xmin)
        ymin = max(-max_limit, ymin)
        xmax = min(max_limit, xmax)
        ymax = min(max_limit, ymax)

        # lim=200
        ax.set_xlim(left=xmin, right=xmax)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        ax_static.set_xlim(left=xmin, right=xmax)
        ax_static.set_ylim(bottom=ymin, top=ymax)
        ax_static.set_aspect('equal')

        line, = ax.plot([], [])

        ax_static.plot(self.point_array[point_array_starting_point:].real,
                       self.point_array[point_array_starting_point:].imag)

        def on_q(event):
            if event.key == 'q':
                exit()

        def init():
            for artist in artist_list:
                ax.add_artist(artist)

            line.set_data([], [])
            return itertools.chain([line], artist_list)

        def get_frames():
            for i in range(self.point_array.size - point_array_starting_point):
                point = i * speed
                if point < self.point_array.size - point_array_starting_point:
                    yield point

        def animate(i):

            for obj in obj_needed_for_drawing:
                obj.update_drawing_objects(i + point_array_starting_point)

            line.set_data(self.point_array[point_array_starting_point:point_array_starting_point + i + 1].real,
                          self.point_array[point_array_starting_point:point_array_starting_point + i + 1].imag)

            return itertools.chain([line], artist_list)

        fig.canvas.mpl_connect('key_press_event', on_q)
        ani = animation.FuncAnimation(fig, animate, interval=40, frames=get_frames,
                                      blit=False,
                                      save_count=1,
                                      init_func=init)

        # export_gcode(drawer)

        plt.show()


class BarMateSlide:
    pass


class BarMateFix:

    def __init__(self):
        self.mate_point_array: Type[Union[np.ndarray, None]] = None

    def get_circle_intersection_with_mate(self, base_points_list: RotationResolution) -> None:
        raise NotImplementedError

    def get_parent_points(self, base_points_list: RotationResolution) -> None:
        raise NotImplementedError


class Anchor(Anchorable, BarMateSlide):

    def __init__(self, complex_number: complex = 0 + 0j):
        super().__init__()
        self.point = complex_number

        """
        Drawing Objects
        """
        self.main_marker = plt.Line2D(
            [self.point.real], [self.point.imag], marker="^")
        self.secondary_marker = plt.Line2D(
            [self.point.real], [self.point.imag], marker="^")

    def create_point_lists(self, base_points_list: RotationResolution):
        self.point_array = np.full_like(
            base_points_list.radial_steps_list, self.point, dtype=complex)
        # self.point_array = self.point

    def update_drawing_objects(self, frame) -> None:
        pass  # do nothing, point stays still

    def get_main_drawing_objects(self) -> List:
        return [self.main_marker, ]

    def get_secondary_drawing_objects(self) -> List:
        return [self.secondary_marker, ]

    def get_min_max_values(self, buffer: float = 0, point_array_only: bool = False, point_array_starting_point: int = 0) -> Tuple:
        return self.point.real - buffer, self.point.real + buffer, self.point.imag - buffer, self.point.imag + buffer

    def get_min_max_values_normalized_to_origin(self, buffer: float = 0) -> Tuple:
        return 0 - buffer, 0 + buffer, 0 - buffer, 0 + buffer

    def get_parent_tree(self) -> List:
        return [self, ]

    def __add__(self, other):
        return Anchor(self.point + other.point)


class Circle(Anchorable, BarMateSlide):
    # exp_const = math.tau * 1j

    def __init__(self,
                 radius: float,
                 frequency: float,
                 starting_angle: float = math.tau,
                 deg: bool = False,
                 parent: Anchorable = Anchor(0 + 0j)):
        # circle defined by C * exp( 2*pi*j * freq * resolution_ticker )
        # C = length * exp( starting_angle * j )
        self.rotation_frequency: float = frequency

        self.radius: float = radius

        if not deg:
            self.starting_angle: float = starting_angle
        else:
            self.starting_angle: float = math.radians(starting_angle)

        # constant
        # self.circle_constants: complex = self.radius * np.exp(self.starting_angle * 1j * self.exp_const * self.rotation_frequency)

        self.parent: Type[Anchorable] = parent

        self.color = random.choice(color_hex)

        """
        Drawing Objects
        """
        self.main_circle_edge_artist = plt.Circle(
            (0, 0), self.radius, fill=False, edgecolor=self.color)
        self.main_centre2point_line_artist = plt.Line2D(
            [], [], marker='.', markevery=(1, 1), color=self.color)

        self.secondary_circle_edge_artist = plt.Circle(
            (0, 0), self.radius, fill=False)
        self.secondary_centre2point_line_artist = plt.Line2D(
            [], [], marker='.', markevery=(1, 1))

        super().__init__()

    def create_point_lists(self, resolution: RotationResolution) -> None:
        if self.point_array is None:
            self.parent.create_point_lists(resolution)
            # parent points + radius * exp(i * angle)
            # angle = starting angle + (rotational multiplier * base rotation)
            self.point_array = self.parent.point_array + (self.radius * np.exp(1j * (
                self.starting_angle + (self.rotation_frequency * resolution.radial_steps_list))))

    def update_drawing_objects(self, frame) -> None:
        # MAIN
        self.main_circle_edge_artist.set_center(
            (self.parent.point_array[frame].real, self.parent.point_array[frame].imag))
        self.main_centre2point_line_artist.set_data(
            [self.parent.point_array[frame].real,
                self.point_array[frame].real],  # x
            [self.parent.point_array[frame].imag, self.point_array[frame].imag])  # y

        # SECONDARY
        self.secondary_centre2point_line_artist.set_data(
            [0, self.point_array[frame].real -
                self.parent.point_array[frame].real],  # x
            [0, self.point_array[frame].imag - self.parent.point_array[frame].imag])  # y

    def get_main_drawing_objects(self) -> List:
        return [self.main_circle_edge_artist, self.main_centre2point_line_artist]

    def get_secondary_drawing_objects(self) -> List:
        return [self.secondary_circle_edge_artist, self.secondary_centre2point_line_artist]

    def get_min_max_values(self, buffer: float = 0, point_array_only: bool = False, point_array_starting_point: int = 0) -> Tuple:
        return super().get_min_max_values(buffer=buffer, point_array_only=point_array_only, point_array_starting_point=point_array_starting_point)

    def get_min_max_values_normalized_to_origin(self, buffer=0) -> Tuple:
        x_max = self.radius + buffer
        x_min = -1 * x_max

        y_max = x_max
        y_min = x_min

        return x_min, x_max, y_min, y_max

    def get_parent_tree(self) -> List:
        return super().get_parent_tree()

    def __str__(self):
        return f'radius: {self.radius}, freq: {self.rotation_frequency}, angle: {self.starting_angle}'


class Spiral(Anchorable, BarMateSlide):
    """
    archimedean:
        parameters:
            0: radius multiplier modifier (bigger the number, the larger the gap between rings)
               (a*phi)

    hyperbolic:
        parameters:
            0: alpha (radius modifier a/phi)

    fermat:
        parameters:
            0: alpha (radius modifier a * phi ^ 1/2)

    lituus:
        parameters:
            0: alpha (radius modifer a / phi ^ 1/2 OR a * phi ^ -1/2)

    logarithmic:
        parameters:
            0: alpha (radius modifer)
            1: ki (radius growth acceleration)
                a * exp(ki * phi)

    poinsots-csch:

    poinsots-sech:

    doppler:


    see: https://en.wikipedia.org/wiki/Spiral
         https://en.wikipedia.org/wiki/List_of_spirals

    """
    spiral_type = ["archimedean", "hyperbolic", "fermat",
                   "lituus", "logarithmic", "poinsots-csch", "poinsots-sech", "doppler"]
    exp_const = math.tau * 1j

    def __init__(self,
                 parent: Type[Anchorable],
                 starting_angle: float,
                 frequency: float,  # bigger number = tigher loops & faster spin
                 type: str,
                 spiral_parameters: List[float],
                 deg: bool = False,
                 ):

        super().__init__()
        self.parent: Type[Anchorable] = parent

        if not deg:
            self.starting_angle: float = starting_angle
        else:
            # convert from deg to radians for internal math
            self.starting_angle: float = math.radian(starting_angle)

        self.rotation_frequency: float = frequency

        if type not in self.spiral_type:
            raise SpiralTypeError
        self.type: str = type

        self.spiral_parameters: List[float] = spiral_parameters

        self.color = random.choice(color_hex)

        """
        Drawing Objects
        """
        self.drawing_object_spiral = plt.Line2D(
            # [], [], marker='*', markevery=(1, 1), linestyle='--')
            [], [], linestyle='--')
        self.drawing_object_parent2spiral = plt.Line2D(
            # [], [], marker='x', markevery=(1, 1), linestyle='-.')
            [], [], marker='x', markevery=None, linestyle='-.')

    def create_point_lists(self, resolution: RotationResolution) -> None:
        if self.point_array is None:
            self.parent.create_point_lists(resolution)

        if self.type == "archimedean":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] * resolution.radial_steps_list) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.radial_steps_list)))
        elif self.type == "hyperbolic":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] / resolution.radial_steps_list) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.radial_steps_list)))
        elif self.type == "fermat":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] * np.sqrt(resolution.radial_steps_list)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.radial_steps_list)))
        elif self.type == "lituus":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] / np.sqrt(resolution.radial_steps_list)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.radial_steps_list)))
        elif self.type == "logarithmic":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] * np.exp(self.spiral_parameters[1] * resolution.radial_steps_list)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.radial_steps_list)))
        elif self.type == "poinsots-csch":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] / np.sinh(self.spiral_parameters[1] * resolution.radial_steps_list)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.radial_steps_list)))
        elif self.type == "poinsots-sech":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] / np.cosh(self.spiral_parameters[1] * resolution.radial_steps_list)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.radial_steps_list)))
        elif self.type == "doppler":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] * ((self.rotation_frequency * resolution.radial_steps_list) * np.cos(self.rotation_frequency * resolution.radial_steps_list) + self.spiral_parameters[1] *  (self.rotation_frequency * resolution.radial_steps_list))) + 1j*(self.spiral_parameters[0] * (self.rotation_frequency * resolution.radial_steps_list) * np.sin(self.rotation_frequency * resolution.radial_steps_list))
                # here lies some happy accidents
                # adding additional rotation to doppler spirals
                # ! DO NOT DELETE
                # np.sqrt(
                #     np.power(self.spiral_parameters[0] * (resolution.radial_steps_list * np.cos(resolution.radial_steps_list) + (self.spiral_parameters[1] * resolution.radial_steps_list)), 2) +
                #     np.power(self.spiral_parameters[0] * resolution.radial_steps_list * np.sin(
                #         resolution.radial_steps_list), 2)
                # ) * np.exp(1j * (self.starting_angle +
                #                  (self.rotation_frequency * resolution.radial_steps_list)))

    def update_drawing_objects(self, frame) -> None:
        # spiral drawing obj
        self.drawing_object_spiral.set_data(
            self.point_array[:frame].real, self.point_array[:frame].imag)

        # center to edge of spriral obj
        self.drawing_object_parent2spiral.set_data([self.parent.point_array[frame].real, self.point_array[frame].real],
                                                   [self.parent.point_array[frame].imag, self.point_array[frame].imag])

    def get_main_drawing_objects(self) -> List:
        return [self.drawing_object_spiral, self.drawing_object_parent2spiral]

    def get_min_max_values(self, buffer: float = 0, point_array_only: bool = False, point_array_starting_point: int = 0) -> Tuple:
        return super().get_min_max_values(buffer=buffer, point_array_only=point_array_only, point_array_starting_point=point_array_starting_point)

    def get_parent_tree(self) -> List:
        return super().get_parent_tree()


class Bar(Anchorable, BarMateFix):

    def __init__(self,
                 parent: Type[Anchorable],
                 point_length_from_parent: float,
                 mate_object: Type[Union[Anchorable, BarMateSlide, BarMateFix]],
                 mate_length_from_parent: float = 0,
                 arm_length: float = 0,
                 arm_angle: float = -np.pi / 2,
                 deg: bool = False):

        super().__init__()
        self.parent: Type[Anchorable] = parent
        self.mate: Type[Union[Anchorable,
                              BarMateSlide, BarMateFix]] = mate_object
        self.point_length = point_length_from_parent
        self.mate_length = mate_length_from_parent
        self.arm_length: float = arm_length
        self.arm_angle: float = arm_angle
        self.mate_point_array = None
        if deg:
            self.arm_angle = np.radians(self.arm_angle)

        self.pre_arm_point_array: Type[Union[np.ndarray, None]] = None

        """
        Drawing Objects
        """
        self.main_mate_line = plt.Line2D(
            [], [], marker='D', markevery=(1, 1), linestyle='--')
        self.main_pre_arm_point_line = plt.Line2D(
            [], [], marker='x', markevery=(1, 1))
        self.main_arm_line = plt.Line2D([], [], marker='.', markevery=(1, 1))
        # self.mate_intersection_circle = plt.Circle(
        # (0, 0), self.mate_length, fill=False)

        self.secondary_mate_line = plt.Line2D(
            [], [], marker='D', markevery=(1, 1), linestyle='--')
        self.secondary_pre_arm_point_line = plt.Line2D(
            [], [], marker='x', markevery=(1, 1))
        self.secondary_arm_line = plt.Line2D(
            [], [], marker='x', markevery=(1, 1))

    def get_parent_points(self, base_points_list: RotationResolution) -> None:
        self.parent.create_point_lists(base_points_list)

    def create_point_lists(self, base_points_list: RotationResolution) -> None:
        if self.point_array is None:
            self.get_parent_points(base_points_list)

            #  mate is a Mate Fix (such as a bar)
            if isinstance(self.mate, BarMateFix):
                self.get_circle_intersection_with_mate(base_points_list)
            # mate is Mate Slide (such as a Circle)
            elif isinstance(self.mate, BarMateSlide):
                self.mate.create_point_lists(base_points_list)
                self.mate_point_array = self.mate.point_array

            # test = np.angle(self.mate_point_array - self.parent.point_array) * 1j

            self.pre_arm_point_array = self.parent.point_array + (
                self.point_length * np.exp(np.angle(self.mate_point_array - self.parent.point_array) * 1j))

            self.point_array = self.pre_arm_point_array + (self.arm_length * np.exp(
                (np.angle(self.mate_point_array - self.parent.point_array) + self.arm_angle) * 1j))
            # self.point_length * np.exp(cmath.phase(self.mate_point_array - self.parent.point_array) * 1j))

    def update_drawing_objects(self, frame) -> None:
        # MAIN
        self.main_mate_line.set_data(
            [self.parent.point_array[frame].real,
                self.mate_point_array[frame].real],
            [self.parent.point_array[frame].imag,
                self.mate_point_array[frame].imag]
        )
        self.main_pre_arm_point_line.set_data(
            [self.parent.point_array[frame].real,
                self.pre_arm_point_array[frame].real],
            [self.parent.point_array[frame].imag,
                self.pre_arm_point_array[frame].imag]
        )
        self.main_arm_line.set_data(
            [self.pre_arm_point_array[frame].real, self.point_array[frame].real],
            [self.pre_arm_point_array[frame].imag, self.point_array[frame].imag],
        )

        # self.mate_intersection_circle.set_center(
        #     (self.parent.point_array[frame].real, self.parent.point_array[frame].imag))

        # SECONDARY
        self.secondary_mate_line.set_data(
            [0, self.mate_point_array[frame].real -
                self.parent.point_array[frame].real],
            [0, self.mate_point_array[frame].imag -
                self.parent.point_array[frame].imag]
        )
        self.secondary_pre_arm_point_line.set_data(
            [0, self.pre_arm_point_array[frame].real -
                self.parent.point_array[frame].real],
            [0, self.pre_arm_point_array[frame].imag -
                self.parent.point_array[frame].imag]
        )
        self.secondary_arm_line.set_data(
            [self.pre_arm_point_array[frame].real - self.parent.point_array[frame].real,
             self.point_array[frame].real - self.parent.point_array[frame].real],
            [self.pre_arm_point_array[frame].imag - self.parent.point_array[frame].imag,
             self.point_array[frame].imag - self.parent.point_array[frame].imag],
        )

    def get_main_drawing_objects(self) -> List:
        # return [self.main_pre_arm_point_line, self.main_mate_line, self.main_arm_line, self.mate_intersection_circle]
        return [self.main_pre_arm_point_line, self.main_mate_line, self.main_arm_line]

    def get_secondary_drawing_objects(self) -> List:
        return [self.secondary_pre_arm_point_line, self.secondary_mate_line, self.secondary_arm_line]

    def get_min_max_values(self, buffer: float = 0, point_array_only: bool = False, point_array_starting_point: int = 0) -> Tuple:

        real_list = [self.point_array[point_array_starting_point:].real]
        imag_list = [self.point_array[point_array_starting_point:].imag]
        if not point_array_only:
            real_list.append(
                self.mate_point_array[point_array_starting_point:].real)
            real_list.append(
                self.parent.point_array[point_array_starting_point:].real)
            real_list.append(
                self.pre_arm_point_array[point_array_starting_point:].real)

            imag_list.append(
                self.mate_point_array[point_array_starting_point:].imag)
            imag_list.append(
                self.parent.point_array[point_array_starting_point:].imag)
            imag_list.append(
                self.pre_arm_point_array[point_array_starting_point:].imag)

        x_min = min(itertools.chain.from_iterable(real_list)) - buffer
        x_max = max(itertools.chain.from_iterable(real_list)) + buffer

        y_min = min(itertools.chain.from_iterable(imag_list)) - buffer
        y_max = max(itertools.chain.from_iterable(imag_list)) + buffer

        return x_min, x_max, y_min, y_max

    # ? should parent be normalized or another point? MATE?
    def get_min_max_values_normalized_to_origin(self, buffer: float = 0) -> Tuple:
        x_min = min(itertools.chain(self.point_array[:].real - self.parent.point_array[:].real,
                                    self.mate_point_array[:].real -
                                    self.parent.point_array[:].real,
                                    self.pre_arm_point_array[:].real -
                                    self.parent.point_array[:].real,
                                    [0])) - buffer
        x_max = max(itertools.chain(self.point_array[:].real - self.parent.point_array[:].real,
                                    self.mate_point_array[:].real -
                                    self.parent.point_array[:].real,
                                    self.pre_arm_point_array[:].real -
                                    self.parent.point_array[:].real,
                                    [0])) + buffer

        y_min = min(itertools.chain(self.point_array[:].imag - self.parent.point_array[:].imag,
                                    self.mate_point_array[:].imag -
                                    self.parent.point_array[:].imag,
                                    self.pre_arm_point_array[:].imag -
                                    self.parent.point_array[:].imag,
                                    [0])) - buffer
        y_max = max(itertools.chain(self.point_array[:].imag - self.parent.point_array[:].imag,
                                    self.mate_point_array[:].imag -
                                    self.parent.point_array[:].imag,
                                    self.pre_arm_point_array[:].imag -
                                    self.parent.point_array[:].imag,
                                    [0])) + buffer

        return x_min, x_max, y_min, y_max

    def get_circle_intersection_with_mate(self, base_points_list: RotationResolution) -> None:
        if self.mate_point_array is None:
            self.mate.get_parent_points(base_points_list)

            x0 = self.parent.point_array.real
            y0 = self.parent.point_array.imag
            r0 = self.mate_length

            x1 = self.mate.parent.point_array.real
            y1 = self.mate.parent.point_array.imag
            r1 = self.mate.mate_length

            if x0[0] > x1[0]:
                x0, y0, r0, x1, y1, r1 = x1, y1, r1, x0, y0, r0

            d = np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))

            a = (np.power(r0, 2) - np.power(r1, 2) + np.power(d, 2)) / (2 * d)
            h = np.sqrt(np.power(r0, 2) - np.power(a, 2))

            x2 = x0 + (a * (x1 - x0)) / d
            y2 = y0 + (a * (y1 - y0)) / d

            x3 = x2 + (h * (y1 - y0)) / d
            y3 = y2 - (h * (x1 - x0)) / d

            x4 = x2 - (h * (y1 - y0)) / d
            y4 = y2 + (h * (x1 - x0)) / d

            self.mate_point_array = x4 + (y4 * 1j)
            self.mate.mate_point_array = self.mate_point_array

            # doing the below calc in fewer varialbes to save on memory
            # a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
            # h = sqrt(r0 ** 2 - a ** 2)
            # x2 = x0 + a * (x1 - x0) / d
            # y2 = y0 + a * (y1 - y0) / d
            # x3 = x2 + h * (y1 - y0) / d
            # y3 = y2 - h * (x1 - x0) / d

        elif self.mate.mate_point_array is None:
            self.mate.mate_point_array = self.mate_point_array

    def get_parent_tree(self) -> List:
        # get parent's tree
        parent_parent_tree = self.parent.get_parent_tree()
        # get mate's tree
        mate_parent_tree = self.mate.get_parent_tree()
        # concatencate mate and parent tree lists
        parent_tree = itertools.chain(
            parent_parent_tree, mate_parent_tree, [self])
        # parent_parent_tree.extend(mate_parent_tree)
        # add self to list
        # parent_tree.append(self)
        # remove duplicate objects from list
        parent_tree = remove_duplicates_from_list(parent_tree)

        return parent_tree


def export_gcode(drawer: Type[Anchorable]):
    with open('/home/dev/drawing_machine/python/cycloid_custom.gcode', 'w') as file_writer:
        file_writer.write("G21 ; mm-mode\n")
        file_writer.write("M3S0\n")
        for coordinate in drawer.point_array:
            x = coordinate.real + 100
            y = coordinate.imag + 100
            file_writer.write(f'G1 F1000 X{x:.2f} Y{y:.2f} Z0\n')

        file_writer.write("M3S255\n")


# base_points = RotationResolution(rotations=40, step_size=0.0005, balanced_around_zero=True)
base_points = RotationResolution(rotations=40, step_size=0.0005)
h = 100
w = 45

c1_size = (h+w)/2
# c2_size = (h-w)/2
c2_size = 40

print(f'c1_size: {c1_size}')
print(f'c2_size: {c2_size}')

a1 = Anchor(0+0j)
# s1 = Spiral(a1, 0, 0.1, "archimedean", [0.01])
# s1 = Spiral(a1, 0, 0.3, "hyperbolic", [10000])
# s1 = Spiral(a1, 0, 0.1, "fermat", [1])
# s1 = Spiral(a1, 0, 0.1, "lituus", [100])
# s1 = Spiral(a1, 0, 0.1, "logarithmic", [10, 0.05])
# s1 = Spiral(a1, 0, 0.1, "poinsots-csch", [1, 0.0001])
# s1 = Spiral(a1, 0, 0.1, "poinsots-sech", [10, 0.04])
s1 = Spiral(a1, 0, 1, "doppler", [1, 0.6])
c1 = Circle(c1_size, 1, starting_angle=np.pi/4, parent=s1)
# c2 = Circle (c2_size, -1, parent=c1)
c2 = Circle(c2_size, -2, parent=c1)
c3 = Circle(c1_size, 3, starting_angle=np.pi, parent=c2)
c4 = Circle(c2_size, -2, parent=c3)
# a1 = Anchor(50+50j)
# b1 = Bar(c4, 70, a1)
# c5 = Circle10, 10, parent=c4)

# b1.animate(base_points, speed=10)
# s1.animate(base_points, speed=100, point_array_starting_point=3000)
s1.animate(base_points, speed=100)

# TODO
# ? is there a better way to chain elements together?
# ?     maybe make a way to build common objects easier
# ? bar needs a better way of connecting two bars together.
# ? create an arc or line obj
# ? create an elipse obj (two circles)
# ? allow all objects to change size
# ? SPIRALS
# ?   implement other spiral types from wiki page
# ?   change rotationaly freq as radian list counts on
# ?   slowing the spin as of the spiral as it gets bigger
# ?   what happens when rotation_freq is a linspace between two numbers?
# ?   how can you start a spiral not from phi = 0 but still sync with the other items?

"""
def animate_full(drawer: Type[Anchorable], resolution_obj: RotationResolution):
    # get full parent tree of drawer (list of obj)
    obj_needed_for_drawing = drawer.get_parent_tree()

    # calculate points needed to draw all objs
    for obj in obj_needed_for_drawing:
        obj.create_point_lists(resolution_obj)

    # get artists for drawing on mpl figure
    artist_list = []

    for obj in obj_needed_for_drawing:
        artist_list.extend(obj.get_main_drawing_objects())

    # add figure and subplots to makes axes
    fig = plt.figure(figsize=[6, 6])
    fig_static = plt.figure(figsize=[6, 6])

    ax = fig.add_subplot(111, frameon=False)
    ax_static = fig_static.add_subplot(111, frameon=False)

    fig.tight_layout()
    fig_static.tight_layout()

    plt.axis('off')

    # get figure limits from drawing objs
    xmin, xmax, ymin, ymax = drawer.get_min_max_values()

    for obj in obj_needed_for_drawing:
        obj_xmin, obj_xmax, obj_ymin, obj_ymax = obj.get_min_max_values()
        xmin = min(xmin, obj_xmin)
        xmax = max(xmax, obj_xmax)
        ymin = min(ymin, obj_ymin)
        ymax = max(ymax, obj_ymax)

    limits_buffer = 5
    xmin = xmin - limits_buffer
    xmax = xmax + limits_buffer
    ymin = ymin - limits_buffer
    ymax = ymax + limits_buffer

    # lim=200
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    ax_static.set_xlim(left=xmin, right=xmax)
    ax_static.set_ylim(bottom=ymin, top=ymax)
    ax_static.set_aspect('equal')

    line, = ax.plot([], [])

    ax_static.plot(drawer.point_array.real, drawer.point_array.imag)

    def on_q(event):
        if event.key == 'q':
            exit()

    def init():
        for artist in artist_list:
            ax.add_artist(artist)

        line.set_data([], [])
        return itertools.chain([line], artist_list)

    def get_frames():
        for i in range(drawer.point_array.size):
            point = i * 1
            if point < drawer.point_array.size:
                yield point

    def animate(i):

        for obj in obj_needed_for_drawing:
            obj.update_drawing_objects(i)

        line.set_data(drawer.point_array[:i + 1].real,
                      drawer.point_array[:i + 1].imag)

        return itertools.chain([line], artist_list)

    fig.canvas.mpl_connect('key_press_event', on_q)
    ani = animation.FuncAnimation(fig, animate, interval=40, frames=get_frames,
                                  blit=False,
                                  save_count=1,
                                  init_func=init)

    # export_gcode(drawer)

    plt.show()

"""


"""
middle_rotation = 9
outer_rotation = -4.1

anchor = Circle(150, 1.1, parent=Anchor(0 +0j))
big_circle = Circle(120*0.5, -3.3 )
circle_middle_circle = Circle(
    20*0.5, middle_rotation, starting_angle=-2*np.pi*(1/8), parent=big_circle)
circle_outside = Circle(54*0.5, outer_rotation, parent=circle_middle_circle)
bar = Bar(anchor, 50.0, circle_outside)
"""

# def animate_all(drawer: Type[Anchorable], resolution_obj: RotationResolution, *components: Type[Anchorable]):
#     """Create point list for drawer and all subcomponents"""
#     drawer.create_point_lists(resolution_obj)

#     for comp in components:
#         comp.create_point_lists(resolution_obj)

#     """Get mpl.artist objects for main drawing"""
#     main_plot_drawer_artist_list = drawer.get_main_drawing_objects()
#     main_plot_component_artist_list = []

#     for comp in components:
#         main_plot_component_artist_list.extend(comp.get_main_drawing_objects())

#     """ Create basic figure and axes """
#     fig = plt.figure(figsize=(21, 13))

#     gs_parent_num_of_rows = 1
#     gs_parent_num_of_cols = 2
#     gs_parent = gridspec.GridSpec(
#         gs_parent_num_of_rows, gs_parent_num_of_cols, figure=fig)

#     num_of_component_plot_rows = 2
#     num_of_component_plot_cols = int(
#         math.ceil(len(components) / num_of_component_plot_rows))

#     print(f'plot component_col: {num_of_component_plot_cols}')

#     gs_component_plots = gridspec.GridSpecFromSubplotSpec(num_of_component_plot_rows, num_of_component_plot_cols,
#                                                           subplot_spec=gs_parent[0])

#     gs_main_plot_num_of_rows = 1
#     gs_main_plot_num_of_cols = 2
#     gs_main_plots = gridspec.GridSpecFromSubplotSpec(gs_main_plot_num_of_rows, gs_main_plot_num_of_cols,
#                                                      subplot_spec=gs_parent[1])

#     main_animated_draw_axis = fig.add_subplot(gs_main_plots[:, 0])
#     main_final_shape_axis = fig.add_subplot(gs_main_plots[:, 1])

#     """ min/max for main drawer axes """
#     animated_space_buffer = 1
#     animated_drawer_x_min, animated_drawer_x_max, animated_drawer_y_min, animated_drawer_y_max = drawer.get_min_max_values(
#         buffer=animated_space_buffer,
#         point_array_only=False)
#     final_space_buffer = 0.2
#     final_drawer_x_min, final_drawer_x_max, final_drawer_y_min, final_drawer_y_max = drawer.get_min_max_values(
#         buffer=final_space_buffer,
#         point_array_only=True)

#     # set xy limits for final drawing axes
#     main_animated_draw_axis.set_xlim(
#         (animated_drawer_x_min, animated_drawer_x_max))
#     main_animated_draw_axis.set_ylim(
#         (animated_drawer_y_min, animated_drawer_y_max))

#     main_final_shape_axis.set_xlim((final_drawer_x_min, final_drawer_x_max))
#     main_final_shape_axis.set_ylim((final_drawer_y_min, final_drawer_y_max))

#     # set axis aspect ratio for final drawing axes
#     main_animated_draw_axis.set_aspect('equal')
#     main_final_shape_axis.set_aspect('equal')

#     main_animated_drawer_final_line, = main_animated_draw_axis.plot([], [])
#     # finished_final_line, = main_final_shape_axis.plot(drawer.point_array.real, drawer.point_array.imag)
#     main_final_shape_axis.plot(
#         drawer.point_array.real, drawer.point_array.imag)

#     # ? What's going on here?
#     comp_axis_artist_list = []
#     individual_axis_space_buffer = 0.5

#     for num, comp in enumerate(components):
#         component_col = int(math.floor(num / num_of_component_plot_rows))
#         component_row = num % num_of_component_plot_rows

#         component_x_min, component_x_max, component_y_min, component_y_max = comp.get_min_max_values_normalized_to_origin(
#             buffer=individual_axis_space_buffer)

#         temp_ax = fig.add_subplot(
#             gs_component_plots[component_row, component_col])
#         temp_ax.set_xlim((component_x_min, component_x_max))
#         temp_ax.set_ylim((component_y_min, component_y_max))
#         temp_ax.set_aspect('equal')

#         for artist in comp.get_secondary_drawing_objects():
#             # add artist to axix
#             temp_ax.add_artist(artist)
#             # add artist to component artist list for figure (needed for animation)
#             comp_axis_artist_list.append(artist)

#     fig.tight_layout()

#     def on_q(event):
#         if event.key == 'q':
#             exit()

#     def init():
#         for artist in itertools.chain(main_plot_drawer_artist_list, main_plot_component_artist_list):
#             main_animated_draw_axis.add_artist(artist)

#         return itertools.chain([main_animated_drawer_final_line],
#                                main_plot_drawer_artist_list,
#                                main_plot_component_artist_list,
#                                comp_axis_artist_list)

#     def get_frames():
#         for i in range(drawer.point_array.size):
#             point = i * 1
#             if point < drawer.point_array.size:
#                 yield point

#     def animate(frame):

#         drawer.update_drawing_objects(frame)

#         for anchorable_obj in components:
#             anchorable_obj.update_drawing_objects(frame)
#         # frame = frame - 1
#         main_animated_drawer_final_line.set_data(drawer.point_array[:frame + 1].real,
#                                                  drawer.point_array[:frame + 1].imag)

#         return itertools.chain([main_animated_drawer_final_line], main_plot_drawer_artist_list,
#                                main_plot_component_artist_list,
#                                comp_axis_artist_list)

#     cid = fig.canvas.mpl_connect('key_press_event', on_q)

#     mpl_animation.FuncAnimation(fig, animate,
#                                 init_func=init,
#                                 interval=1,
#                                 blit=True,
#                                 save_count=1,
#                                 frames=get_frames,
#                                 repeat=False)

#     plt.show()
