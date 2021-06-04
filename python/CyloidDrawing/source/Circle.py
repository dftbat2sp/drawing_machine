# import cmath
import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Type, Union, Dict
import random

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


class IntersectionError(Exception):
    pass


class SpiralTypeError(Exception):
    pass


class ListOfPointsParameterError(Exception):
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

color_hex = [
    '#264653', '#2a9d8f', '#e76f51', '#e76f51', '#e63946', '#06d6a0',
    '#118ab2', '#f4acb7', '#8338ec', '#9a031e', '#fb5607', '#22223b',
    '#3a0ca3', '#3a0ca3'
]


def remove_duplicates_from_list(my_list):
    return list(dict.fromkeys(my_list))


# @dataclass
# class RotationResolution:
#     step_size: float = 0.001
#     rotations: float = 50.0
#     balanced_around_zero: bool = False

#     def __post_init__(self):
#         # self.num_of_points: int = int(self.rotations / self.step_size)
#         self.num_of_rotation_in_radians: float = self.rotations * math.tau
#         # add +1 to make divisions of a rotation make sense
#         # before step size of tau/3 for 1 rotation would make a list of 0, pi, tau
#         # now it makes this: 0, tau/3, 2*tau/3, tau
#         self.num_of_points: int = int(self.num_of_rotation_in_radians /
#                                       self.step_size)
#         if self.balanced_around_zero:
#             half_rotation_in_radian = self.num_of_rotation_in_radians / 2
#             self.point_array: np.ndarray = np.linspace(
#                 -half_rotation_in_radian, half_rotation_in_radian,
#                 self.num_of_points)
#         else:
#             self.point_array: np.ndarray = np.linspace(
#                 # 0, self.num_of_rotation_in_radians, self.num_of_points)
#                 self.step_size,
#                 self.num_of_rotation_in_radians,
#                 self.num_of_points)


class ListOfPoints:
    """
    ListOfPoints
    Create a list of numbers
    ---
    kwargs:
        distance_between_points: float
        num_of_rotations: float
        num_of_points: int

    combos:
        (2) step size + total points
        (1) step size + num of rotations (stops after first point > then rotation)
        (3) num of rotations + points per rotation
        (4) num of rotations + total points
    """
    def __init__(self,
                 step_size: float = None,
                 total_num_of_points: int = None,
                 num_of_rotations: float = None,
                 num_of_points_per_rotation: int = None,
                 balanced_around_zero: bool = False):
        self.point_array = None

        if total_num_of_points is not None:
            total_num_of_points = math.ceil(total_num_of_points)

        if num_of_points_per_rotation is not None:
            num_of_points_per_rotation = math.ceil(num_of_points_per_rotation)

        if num_of_rotations is not None:
            num_of_rotations_in_radians = num_of_rotations * math.tau

        # ensure total num of points is an int
        if step_size is not None and num_of_rotations is not None:
            num_of_steps = math.ceil(num_of_rotations_in_radians / step_size)
            if balanced_around_zero:
                # ensure num_of_steps is even
                num_of_steps = num_of_steps + (num_of_steps % 2)
                half_num_of_points = num_of_steps / 2
                self.point_array = np.arange(
                    -half_num_of_points, half_num_of_points + 1) * step_size
            else:
                self.point_array = np.arange(0, num_of_steps + 1) * step_size

        elif step_size is not None and total_num_of_points is not None:
            if balanced_around_zero:
                half_num_of_points = total_num_of_points + \
                    (total_num_of_points % 2)
                self.point_array = np.arange(
                    -half_num_of_points, half_num_of_points + 1) * step_size
            else:
                self.point_array = np.arange(
                    0, total_num_of_points + 1) * step_size

        elif num_of_rotations is not None and num_of_points_per_rotation is not None:
            num_of_points = num_of_rotations * num_of_points_per_rotation
            if balanced_around_zero:
                half_num_of_radians = num_of_rotations_in_radians / 2
                self.point_array = np.linspace(-half_num_of_radians,
                                               half_num_of_radians,
                                               num_of_points)
            else:
                # add 1 to account for starting at 0
                self.point_array = np.linspace(0, num_of_rotations_in_radians,
                                               num_of_points + 1)

        elif num_of_rotations is not None and total_num_of_points is not None:
            if balanced_around_zero:
                half_num_of_radians = num_of_rotations_in_radians / 2
                self.point_array = np.linspace(-half_num_of_radians,
                                               half_num_of_radians,
                                               total_num_of_points)
            else:
                # add 1 to account for starting at 0
                self.point_array = np.linspace(0, num_of_rotations_in_radians,
                                               total_num_of_points + 1)
        else:
            raise ListOfPointsParameterError


class Anchorable:
    def __init__(self):
        self.point_array: Type[Union[np.ndarray, None]] = None
        # self.list_calculated = False

    def create_point_lists(self, foundation_list_of_points: RotationResolution) -> None:
        raise NotImplementedError

    def update_drawing_objects(self, frame) -> None:
        raise NotImplementedError

    def get_main_drawing_objects(self) -> List:
        raise NotImplementedError

    def get_secondary_drawing_objects(self) -> List:
        raise NotImplementedError

    def get_min_max_values(self,
                           buffer: float = 0,
                           point_array_only: bool = False,
                           point_array_starting_offset: int = 0) -> Tuple:
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
            self.point_array[point_array_starting_offset:].real) - buffer
        x_max = max(
            self.point_array[point_array_starting_offset:].real) + buffer

        y_min = min(
            self.point_array[point_array_starting_offset:].imag) - buffer
        y_max = max(
            self.point_array[point_array_starting_offset:].imag) + buffer

        return x_min, x_max, y_min, y_max

    def get_min_max_values_normalized_to_origin(self,
                                                buffer: float = 0) -> Tuple:
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

    def export_gcode(self,
                     resolution_obj: RotationResolution,
                     canvas_xlimit: int = 330,
                     canvas_ylimit: int = 330,
                     canvas_axes_buffer: int = 10,
                     point_array_starting_offset: int = 0,
                     decimal_precision: int = 1,
                     file_name: str = "spiral"):

        # ! fix scaling so that images aren't warped
        # i.e. squished in x or y direction
        self.create_point_lists(resolution_obj)

        with open(f'C:\\Users\\mealy\\Desktop\\gcode\\{file_name}.gcode',
                  'w') as file_writer:
            file_writer.write("G21 ; mm-mode\n")
            file_writer.write("M3S0\n")
            previous_coordinate = 0 + 0j
            for coordinate in self._scale_point_array(
                    canvas_xlimit, canvas_ylimit, canvas_axes_buffer,
                    point_array_starting_offset):
                # x = coordinate.real
                # y = coordinate.imag
                if f'{coordinate:.1f}' != f'{previous_coordinate:.1f}':
                    file_writer.write(
                        f'G1 F5000 X{coordinate.real:.{decimal_precision}f} Y{coordinate.imag:.{decimal_precision}f} Z0\n'
                    )
                previous_coordinate = coordinate

            file_writer.write("M3S255\n")

    def _scale_point_array(self,
                           canvas_xlimit: int,
                           canvas_ylimit: int,
                           canvas_axes_buffer: int,
                           point_array_starting_offset=0) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.get_min_max_values(
            point_array_starting_offset=point_array_starting_offset)
        # amount of space to draw is limit - axes buffer
        canvas_xlimit_minus_axis_buffer = canvas_xlimit - canvas_axes_buffer
        canvas_ylimit_minus_axis_buffer = canvas_ylimit - canvas_axes_buffer
        # take the biggest different between the x and y direction
        # if you took them seperately, it would squeeze or stretch the image
        # max_difference_between_points = max((xmax - xmin), (ymax - ymin))
        # print(f'xmax: {xmax}\nxmin: {xmin}\nxdiff: {xmax-xmin}\nymax: {ymax}\nymin: {ymin}\nydiff: {ymax - ymin}\nmax-diff: {max_difference_between_points}')

        # difference between the smallest and largest x coordinate
        max_difference_in_x_direction = xmax - xmin
        # difference between teh smallest and largest y coordinate
        max_difference_in_y_direction = ymax - ymin

        # how much does x need to scale for the farthest point is scaled to the canvas' limit
        scale_ratio_x_direction = canvas_xlimit_minus_axis_buffer / \
            max_difference_in_x_direction
        # how much does y need to scale for the farthest point is scaled to the canvas' limit
        scale_ratio_y_direction = canvas_ylimit_minus_axis_buffer / \
            max_difference_in_y_direction

        canvas_scale_ratio = min(scale_ratio_x_direction,
                                 scale_ratio_y_direction)

        # shift points so xmin and ymin are on theie respective axes
        scaled_point_array = self.point_array[point_array_starting_offset:]

        scaled_point_array[:].real = scaled_point_array[:].real - xmin
        scaled_point_array[:].imag = scaled_point_array[:].imag - ymin
        # scale all points to inside the canvas limits then add buffer from the axes
        # scaled_point_array[:].real = scaled_point_array[:].real * (canvas_true_x_difference / shape_x_difference) + canvas_axes_buffer
        scaled_point_array[:].real = scaled_point_array[:].real * \
            (canvas_scale_ratio)
        scaled_point_array[:].real = scaled_point_array[:].real + \
            canvas_axes_buffer
        # scaled_point_array[:].imag = scaled_point_array[:].imag * (canvas_true_y_difference / shape_y_difference) + canvas_axes_buffer
        scaled_point_array[:].imag = scaled_point_array[:].imag * \
            (canvas_scale_ratio)
        scaled_point_array[:].imag = scaled_point_array[:].imag + \
            canvas_axes_buffer

        return scaled_point_array

    def animate(self,
                resolution_obj: RotationResolution,
                speed: int = 1,
                point_array_starting_offset: int = 0) -> None:

        # get full parent tree of drawer (list of obj)
        obj_needed_for_drawing = self.get_parent_tree()

        # get our own points list
        self.create_point_lists(resolution_obj)

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
            point_array_starting_offset=point_array_starting_offset)

        for obj in obj_needed_for_drawing:
            obj_xmin, obj_xmax, obj_ymin, obj_ymax = obj.get_min_max_values(
                point_array_starting_offset=point_array_starting_offset)
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

        ax_static.plot(self.point_array[point_array_starting_offset:].real,
                       self.point_array[point_array_starting_offset:].imag)

        def on_q(event):
            if event.key == 'q':
                exit()

        def init():
            for artist in artist_list:
                ax.add_artist(artist)

            line.set_data([], [])
            return itertools.chain([line], artist_list)

        def get_frames():
            for i in range(self.point_array.size -
                           point_array_starting_offset):
                point = i * speed
                if point < self.point_array.size - point_array_starting_offset:
                    yield point

        def matplotlib_animate(i):

            for obj in obj_needed_for_drawing:
                obj.update_drawing_objects(i + point_array_starting_offset)

            line.set_data(
                self.point_array[
                    point_array_starting_offset:point_array_starting_offset +
                    i + 1].real,
                self.point_array[
                    point_array_starting_offset:point_array_starting_offset +
                    i + 1].imag)

            return itertools.chain([line], artist_list)

        fig.canvas.mpl_connect('key_press_event', on_q)
        ani = animation.FuncAnimation(fig,
                                      matplotlib_animate,
                                      interval=40,
                                      frames=get_frames,
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

    def get_circle_intersection_with_mate(
            self, foundation_list_of_points: RotationResolution) -> None:
        raise NotImplementedError

    def get_parent_points(self, foundation_list_of_points: RotationResolution) -> None:
        raise NotImplementedError


class Anchor(Anchorable, BarMateSlide):
    def __init__(self, complex_number: complex = 0 + 0j):
        super().__init__()
        self.point = complex_number
        """
        Drawing Objects
        """
        self.main_marker = plt.Line2D([self.point.real], [self.point.imag],
                                      marker="^")
        self.secondary_marker = plt.Line2D([self.point.real],
                                           [self.point.imag],
                                           marker="^")

    def create_point_lists(self, foundation_list_of_points: RotationResolution):
        self.point_array = np.full_like(foundation_list_of_points.point_array,
                                        self.point,
                                        dtype=complex)
        # self.point_array = self.point

    def update_drawing_objects(self, frame) -> None:
        pass  # do nothing, point stays still

    def get_main_drawing_objects(self) -> List:
        return [
            self.main_marker,
        ]

    def get_secondary_drawing_objects(self) -> List:
        return [
            self.secondary_marker,
        ]

    def get_min_max_values(self,
                           buffer: float = 0,
                           point_array_only: bool = False,
                           point_array_starting_offset: int = 0) -> Tuple:
        return self.point.real - buffer, self.point.real + buffer, self.point.imag - buffer, self.point.imag + buffer

    def get_min_max_values_normalized_to_origin(self,
                                                buffer: float = 0) -> Tuple:
        return 0 - buffer, 0 + buffer, 0 - buffer, 0 + buffer

    def get_parent_tree(self) -> List:
        return [
            self,
        ]

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
        self.main_circle_edge_artist = plt.Circle((0, 0),
                                                  self.radius,
                                                  fill=False,
                                                  edgecolor=self.color)
        self.main_centre2point_line_artist = plt.Line2D([], [],
                                                        marker='.',
                                                        markevery=(1, 1),
                                                        color=self.color)

        self.secondary_circle_edge_artist = plt.Circle((0, 0),
                                                       self.radius,
                                                       fill=False)
        self.secondary_centre2point_line_artist = plt.Line2D([], [],
                                                             marker='.',
                                                             markevery=(1, 1))

        super().__init__()

    def create_point_lists(self, resolution: ListOfPoints) -> None:
        if self.point_array is None:
            self.parent.create_point_lists(resolution)
            # parent points + radius * exp(i * angle)
            # angle = starting angle + (rotational multiplier * base rotation)
            self.point_array = self.parent.point_array + (self.radius * np.exp(
                1j *
                (self.starting_angle +
                 (self.rotation_frequency * resolution.point_array))))

    def update_drawing_objects(self, frame) -> None:
        # MAIN
        self.main_circle_edge_artist.set_center(
            (self.parent.point_array[frame].real,
             self.parent.point_array[frame].imag))
        self.main_centre2point_line_artist.set_data(
            [
                self.parent.point_array[frame].real,
                self.point_array[frame].real
            ],  # x
            [
                self.parent.point_array[frame].imag,
                self.point_array[frame].imag
            ])  # y

        # SECONDARY
        self.secondary_centre2point_line_artist.set_data(
            [
                0, self.point_array[frame].real -
                self.parent.point_array[frame].real
            ],  # x
            [
                0, self.point_array[frame].imag -
                self.parent.point_array[frame].imag
            ])  # y

    def get_main_drawing_objects(self) -> List:
        return [
            self.main_circle_edge_artist, self.main_centre2point_line_artist
        ]

    def get_secondary_drawing_objects(self) -> List:
        return [
            self.secondary_circle_edge_artist,
            self.secondary_centre2point_line_artist
        ]

    def get_min_max_values(self,
                           buffer: float = 0,
                           point_array_only: bool = False,
                           point_array_starting_offset: int = 0) -> Tuple:
        return super().get_min_max_values(
            buffer=buffer,
            point_array_only=point_array_only,
            point_array_starting_offset=point_array_starting_offset)

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

    doppler+rotation:


    see: https://en.wikipedia.org/wiki/Spiral
         https://en.wikipedia.org/wiki/List_of_spirals

    """
    spiral_type = [
        "archimedean", "hyperbolic", "fermat", "lituus", "logarithmic",
        "poinsots-csch", "poinsots-sech", "doppler", "doppler+rotation"
    ]
    exp_const = math.tau * 1j

    def __init__(
        self,
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
            [],
            [],
            linestyle='--')
        self.drawing_object_parent2spiral = plt.Line2D(
            # [], [], marker='x', markevery=(1, 1), linestyle='-.')
            [],
            [],
            marker='x',
            markevery=None,
            linestyle='-.')

    def create_point_lists(self, resolution: RotationResolution) -> None:
        if self.point_array is None:
            self.parent.create_point_lists(resolution)

        if self.type == "archimedean":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] * resolution.point_array) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.point_array)))
        elif self.type == "hyperbolic":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] / resolution.point_array) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.point_array)))
        elif self.type == "fermat":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] * np.sqrt(resolution.point_array)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.point_array)))
        elif self.type == "lituus":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] / np.sqrt(resolution.point_array)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.point_array)))
        elif self.type == "logarithmic":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] * np.exp(self.spiral_parameters[1] * resolution.point_array)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.point_array)))
        elif self.type == "poinsots-csch":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] / np.sinh(self.spiral_parameters[1] * resolution.point_array)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.point_array)))
        elif self.type == "poinsots-sech":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] / np.cosh(self.spiral_parameters[1] * resolution.point_array)) * \
                np.exp(1j * (self.starting_angle +
                       (self.rotation_frequency * resolution.point_array)))
        elif self.type == "doppler":
            self.point_array = self.parent.point_array + \
                (self.spiral_parameters[0] * ((self.rotation_frequency * resolution.point_array) * np.cos(self.rotation_frequency * resolution.point_array) + self.spiral_parameters[1] * (
                    self.rotation_frequency * resolution.point_array))) + 1j*(self.spiral_parameters[0] * (self.rotation_frequency * resolution.point_array) * np.sin(self.rotation_frequency * resolution.point_array))
        elif self.type == "doppler+rotation":
            self.point_array = self.parent.point_array + \
                np.sqrt(
                    np.power(self.spiral_parameters[0] * (resolution.point_array * np.cos(resolution.point_array) + (self.spiral_parameters[1] * resolution.point_array)), 2) +
                    np.power(self.spiral_parameters[0] * resolution.point_array * np.sin(
                        resolution.point_array), 2)
                ) * np.exp(1j * (self.starting_angle +
                                 (self.rotation_frequency * resolution.point_array)))

    def update_drawing_objects(self, frame) -> None:
        # spiral drawing obj
        self.drawing_object_spiral.set_data(self.point_array[:frame].real,
                                            self.point_array[:frame].imag)

        # center to edge of spriral obj
        self.drawing_object_parent2spiral.set_data([
            self.parent.point_array[frame].real, self.point_array[frame].real
        ], [self.parent.point_array[frame].imag, self.point_array[frame].imag])

    def get_main_drawing_objects(self) -> List:
        return [self.drawing_object_spiral, self.drawing_object_parent2spiral]

    def get_min_max_values(self,
                           buffer: float = 0,
                           point_array_only: bool = False,
                           point_array_starting_offset: int = 0) -> Tuple:
        return super().get_min_max_values(
            buffer=buffer,
            point_array_only=point_array_only,
            point_array_starting_offset=point_array_starting_offset)

    def get_parent_tree(self) -> List:
        return super().get_parent_tree()


class Bar(Anchorable, BarMateFix):
    def __init__(self,
                 parent: Type[Anchorable],
                 point_length_from_parent: float,
                 mate_object: Type[Union[Anchorable, BarMateSlide,
                                         BarMateFix]],
                 mate_length_from_parent: float = 0,
                 arm_length: float = 0,
                 arm_angle: float = -np.pi / 2,
                 deg: bool = False):

        super().__init__()
        self.parent: Type[Anchorable] = parent
        self.mate: Type[Union[Anchorable, BarMateSlide,
                              BarMateFix]] = mate_object
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
        self.main_mate_line = plt.Line2D([], [],
                                         marker='D',
                                         markevery=(1, 1),
                                         linestyle='--')
        self.main_pre_arm_point_line = plt.Line2D([], [],
                                                  marker='x',
                                                  markevery=(1, 1))
        self.main_arm_line = plt.Line2D([], [], marker='.', markevery=(1, 1))
        # self.mate_intersection_circle = plt.Circle(
        # (0, 0), self.mate_length, fill=False)

        self.secondary_mate_line = plt.Line2D([], [],
                                              marker='D',
                                              markevery=(1, 1),
                                              linestyle='--')
        self.secondary_pre_arm_point_line = plt.Line2D([], [],
                                                       marker='x',
                                                       markevery=(1, 1))
        self.secondary_arm_line = plt.Line2D([], [],
                                             marker='x',
                                             markevery=(1, 1))

    def get_parent_points(self, foundation_list_of_points: RotationResolution) -> None:
        self.parent.create_point_lists(foundation_list_of_points)

    def create_point_lists(self, foundation_list_of_points: RotationResolution) -> None:
        if self.point_array is None:
            self.get_parent_points(foundation_list_of_points)

            #  mate is a Mate Fix (such as a bar)
            if isinstance(self.mate, BarMateFix):
                self.get_circle_intersection_with_mate(foundation_list_of_points)
            # mate is Mate Slide (such as a Circle)
            elif isinstance(self.mate, BarMateSlide):
                self.mate.create_point_lists(foundation_list_of_points)
                self.mate_point_array = self.mate.point_array

            # test = np.angle(self.mate_point_array - self.parent.point_array) * 1j

            self.pre_arm_point_array = self.parent.point_array + (
                self.point_length * np.exp(
                    np.angle(self.mate_point_array - self.parent.point_array) *
                    1j))

            self.point_array = self.pre_arm_point_array + (
                self.arm_length * np.exp(
                    (np.angle(self.mate_point_array - self.parent.point_array)
                     + self.arm_angle) * 1j))
            # self.point_length * np.exp(cmath.phase(self.mate_point_array - self.parent.point_array) * 1j))

    def update_drawing_objects(self, frame) -> None:
        # MAIN
        self.main_mate_line.set_data([
            self.parent.point_array[frame].real,
            self.mate_point_array[frame].real
        ], [
            self.parent.point_array[frame].imag,
            self.mate_point_array[frame].imag
        ])
        self.main_pre_arm_point_line.set_data([
            self.parent.point_array[frame].real,
            self.pre_arm_point_array[frame].real
        ], [
            self.parent.point_array[frame].imag,
            self.pre_arm_point_array[frame].imag
        ])
        self.main_arm_line.set_data(
            [
                self.pre_arm_point_array[frame].real,
                self.point_array[frame].real
            ],
            [
                self.pre_arm_point_array[frame].imag,
                self.point_array[frame].imag
            ],
        )

        # self.mate_intersection_circle.set_center(
        #     (self.parent.point_array[frame].real, self.parent.point_array[frame].imag))

        # SECONDARY
        self.secondary_mate_line.set_data([
            0, self.mate_point_array[frame].real -
            self.parent.point_array[frame].real
        ], [
            0, self.mate_point_array[frame].imag -
            self.parent.point_array[frame].imag
        ])
        self.secondary_pre_arm_point_line.set_data([
            0, self.pre_arm_point_array[frame].real -
            self.parent.point_array[frame].real
        ], [
            0, self.pre_arm_point_array[frame].imag -
            self.parent.point_array[frame].imag
        ])
        self.secondary_arm_line.set_data(
            [
                self.pre_arm_point_array[frame].real -
                self.parent.point_array[frame].real,
                self.point_array[frame].real -
                self.parent.point_array[frame].real
            ],
            [
                self.pre_arm_point_array[frame].imag -
                self.parent.point_array[frame].imag,
                self.point_array[frame].imag -
                self.parent.point_array[frame].imag
            ],
        )

    def get_main_drawing_objects(self) -> List:
        # return [self.main_pre_arm_point_line, self.main_mate_line, self.main_arm_line, self.mate_intersection_circle]
        return [
            self.main_pre_arm_point_line, self.main_mate_line,
            self.main_arm_line
        ]

    def get_secondary_drawing_objects(self) -> List:
        return [
            self.secondary_pre_arm_point_line, self.secondary_mate_line,
            self.secondary_arm_line
        ]

    def get_min_max_values(self,
                           buffer: float = 0,
                           point_array_only: bool = False,
                           point_array_starting_offset: int = 0) -> Tuple:

        real_list = [self.point_array[point_array_starting_offset:].real]
        imag_list = [self.point_array[point_array_starting_offset:].imag]
        if not point_array_only:
            real_list.append(
                self.mate_point_array[point_array_starting_offset:].real)
            real_list.append(
                self.parent.point_array[point_array_starting_offset:].real)
            real_list.append(
                self.pre_arm_point_array[point_array_starting_offset:].real)

            imag_list.append(
                self.mate_point_array[point_array_starting_offset:].imag)
            imag_list.append(
                self.parent.point_array[point_array_starting_offset:].imag)
            imag_list.append(
                self.pre_arm_point_array[point_array_starting_offset:].imag)

        x_min = min(itertools.chain.from_iterable(real_list)) - buffer
        x_max = max(itertools.chain.from_iterable(real_list)) + buffer

        y_min = min(itertools.chain.from_iterable(imag_list)) - buffer
        y_max = max(itertools.chain.from_iterable(imag_list)) + buffer

        return x_min, x_max, y_min, y_max

    # ? should parent be normalized or another point? MATE?
    def get_min_max_values_normalized_to_origin(self,
                                                buffer: float = 0) -> Tuple:
        x_min = min(
            itertools.chain(
                self.point_array[:].real - self.parent.point_array[:].real,
                self.mate_point_array[:].real -
                self.parent.point_array[:].real,
                self.pre_arm_point_array[:].real -
                self.parent.point_array[:].real, [0])) - buffer
        x_max = max(
            itertools.chain(
                self.point_array[:].real - self.parent.point_array[:].real,
                self.mate_point_array[:].real -
                self.parent.point_array[:].real,
                self.pre_arm_point_array[:].real -
                self.parent.point_array[:].real, [0])) + buffer

        y_min = min(
            itertools.chain(
                self.point_array[:].imag - self.parent.point_array[:].imag,
                self.mate_point_array[:].imag -
                self.parent.point_array[:].imag,
                self.pre_arm_point_array[:].imag -
                self.parent.point_array[:].imag, [0])) - buffer
        y_max = max(
            itertools.chain(
                self.point_array[:].imag - self.parent.point_array[:].imag,
                self.mate_point_array[:].imag -
                self.parent.point_array[:].imag,
                self.pre_arm_point_array[:].imag -
                self.parent.point_array[:].imag, [0])) + buffer

        return x_min, x_max, y_min, y_max

    def get_circle_intersection_with_mate(
            self, foundation_list_of_points: RotationResolution) -> None:
        if self.mate_point_array is None:
            self.mate.get_parent_points(foundation_list_of_points)

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

            # x3 = x2 + (h * (y1 - y0)) / d
            # y3 = y2 - (h * (x1 - x0)) / d

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
        parent_tree = itertools.chain(parent_parent_tree, mate_parent_tree,
                                      [self])
        # parent_parent_tree.extend(mate_parent_tree)
        # add self to list
        # parent_tree.append(self)
        # remove duplicate objects from list
        parent_tree = remove_duplicates_from_list(parent_tree)

        return parent_tree


# base_points = RotationResolution(rotations=40, step_size=0.0005, balanced_around_zero=True)
# base_points = RotationResolution(rotations=39.5, step_size=0.0005)
# base_points = RotationResolution(rotations=60, step_size=math.tau/4, balanced_around_zero=True)
# base_points = RotationResolution(rotations=60, step_size=math.tau / 4)
foundation_point_array = ListOfPoints(num_of_rotations=50, step_size=0.005)
h = 100
w = 45

c1_size = (h + w) / 2
# c2_size = (h-w)/2
c2_size = 40

# print(f'c1_size: {c1_size}')
# print(f'c2_size: {c2_size}')

a1 = Anchor(0 + 0j)
s1 = Spiral(a1, 0, 1, "archimedean", [0.3])
# s1 = Spiral(a1, 0, 0.3, "hyperbolic", [10000])
# s1 = Spiral(a1, 0, 0.1, "fermat", [1])
# s1 = Spiral(a1, 0, 0.1, "lituus", [100])
# s1 = Spiral(a1, 0, 0.1, "logarithmic", [10, 0.05])
# s1 = Spiral(a1, 0, 0.1, "poinsots-csch", [1, 0.0001])
# s1 = Spiral(a1, 0, 0.1, "poinsots-sech", [10, 0.04])
# s1 = Spiral(a1, 0, 1, "doppler", [1, 0.6])
# s1 = Spiral(a1, 0, 1.04, "doppler+rotation", [1, 0.7])
c1 = Circle(c1_size, 1, starting_angle=np.pi / 4, parent=s1)
# c2 = Circle (c2_size, -1, parent=c1)
c2 = Circle(c2_size, -2, parent=c1)
c3 = Circle(c1_size, 3, starting_angle=np.pi, parent=c2)
c4 = Circle(c2_size, -2, parent=c3)
# a1 = Anchor(50+50j)
# b1 = Bar(c4, 70, a1)
# c5 = Circle10, 10, parent=c4)

# b1.animate(base_points, speed=10)
# s1.export_gcode(base_points, point_array_starting_offset=0, decimal_precision=1, file_name="spirangle", canvas_xlimit=205, canvas_ylimit=270)
# s1.animate(base_points, speed=100, point_array_starting_offset=1000)
s1.animate(foundation_point_array, speed=1)

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
# ? curve fitting to turn discrete points into G02 G03 arcs
# ?   group discrete lumps of points by constant increasing or decreating of x or y
