# import cmath
import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Type, Union, Dict
import random
from matplotlib import lines

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import sys


class IntersectionError(Exception):
    pass


class SpiralTypeError(Exception):
    pass


class ListOfPointsParameterError(Exception):
    pass


class SpiralParametersError(Exception):
    pass


class PointArrayEmpty(Exception):
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

color_hex = [
    "#00FF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFF00",
    "#000000",
    "#70DB93",
    "#B5A642",
    "#5F9F9F",
    "#B87333",
    "#2F4F2F",
    "#9932CD",
    "#871F78",
    "#855E42",
    "#545454",
    "#8E2323",
    "#F5CCB0",
    "#238E23",
    "#CD7F32",
    "#DBDB70",
    "#C0C0C0",
    "#527F76",
    "#9F9F5F",
    "#8E236B",
    "#2F2F4F",
    "#EBC79E",
    "#CFB53B",
    "#FF7F00",
    "#DB70DB",
    "#D9D9F3",
    "#5959AB",
    "#8C1717",
    "#238E68",
    "#6B4226",
    "#8E6B23",
    "#007FFF",
    "#00FF7F",
    "#236B8E",
    "#38B0DE",
    "#DB9370",
    "#ADEAEA",
    "#5C4033",
    "#4F2F4F",
    "#CC3299",
    "#99CC32",
]
#! TEMP
np.set_printoptions(threshold=sys.maxsize)


def remove_duplicates_from_list(my_list):
    return list(dict.fromkeys(my_list))


# @dataclass
# class ListOfPoints:
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

    num_of_points_max = 0
    lowest_common_multiple_of_num_of_cycles = 1
    final_number_of_points = 0

    def __init__(
            self,
            steps_per_cycle: int = 720,
            #  total_num_of_points: int = None,
            #  num_of_rotations: float = 50,
            #  num_of_points_per_rotation: int = None,
            cycle_start: float = 0,
            cycle_end: float = 1,
            num_of_cycles: int = 1):

        self.point_array: Union[np.ndarray, None] = None

        if not isinstance(steps_per_cycle, int):
            raise ListOfPointsParameterError('steps_per_cycle is not an INT')
        if not isinstance(num_of_cycles, int):
            raise ListOfPointsParameterError('num_of_cycles is not an INT')

        self.num_of_cycles: int = num_of_cycles
        self.cycle_start: float = cycle_start
        self.cycle_end: float = cycle_end

        self.total_cycles: float = np.abs(self.cycle_end - self.cycle_start)

        if self.total_cycles == 0:
            raise ListOfPointsParameterError(
                'cycle_end - cycle_start resulted in zero')

        self.initial_num_of_points: int = self.get_num_of_points(
            self.total_cycles, steps_per_cycle, self.num_of_cycles)

        ListOfPoints.num_of_points_max = max(ListOfPoints.num_of_points_max,
                                             self.initial_num_of_points)

        ListOfPoints.lowest_common_multiple_of_num_of_cycles = np.lcm(
            ListOfPoints.lowest_common_multiple_of_num_of_cycles,
            self.num_of_cycles)

    @staticmethod
    def get_num_of_points(total_cycles: float, steps_per_cycle: int,
                          num_of_cycles: int) -> int:
        # cycles = TAU * total_cycles
        # step_size = TAU/step_size
        # typically for #ofPoints you wants total / step size
        # which would be TAU * cycles / TAU / step size == TAU * cycles * step_size / TAU == cycles * step_size
        # +1 to account for fence post problem | - - - - | 1 cycle from 0-1 with 5 step size
        # 0, t/5, 2t/5, 3t/5, 4t/5, 5t/5 <--- 6 points with 1 cycles 5 step size
        # 2 cycles from 0-1 (0-1-0), 5 step size
        # 0, t/5, 2t/5, 3t/5, 4t/5, 5t/5, 4t/5, 3t/5, 2t/5, t/5, 0 = 11 points = (2 cycles * total_cycles * step size) + 1
        # +1 added in calc common num of points
        return math.ceil(total_cycles * steps_per_cycle * num_of_cycles)

    @staticmethod
    def calculate_common_num_of_points():
        points_mod_lcm = ListOfPoints.num_of_points_max % ListOfPoints.lowest_common_multiple_of_num_of_cycles
        lcm_minus_points_mod_lcm = 0

        # no need to add anything to num of points if points mod lcm is 0
        if not points_mod_lcm == 0:
            lcm_minus_points_mod_lcm = ListOfPoints.lowest_common_multiple_of_num_of_cycles - points_mod_lcm
        # see def get_num_of_points why +1 is added to final_number_of_points
        ListOfPoints.final_number_of_points = ListOfPoints.num_of_points_max + lcm_minus_points_mod_lcm + 1
        # return ListOfPoints.final_number_of_points

    #TODO incorporate # of cycles
    def calculcate_points_array(self) -> None:
        self.point_array = np.linspace(self.cycle_start * math.tau,
                                       self.cycle_end * math.tau,
                                       ListOfPoints.final_number_of_points)


class Anchorable:
    def __init__(self):
        self.point_array: Union[np.ndarray, None] = None
        # self.list_calculated = False

    def create_point_lists(self,
                           foundation_list_of_points: ListOfPoints) -> None:
        raise NotImplementedError

    def update_drawing_objects(self, frame) -> None:
        raise NotImplementedError

    def get_main_drawing_objects(self) -> List:
        raise NotImplementedError

    def get_secondary_drawing_objects(self) -> List:
        raise NotImplementedError

    def get_min_max_values(self,
                           buffer: float = 0,
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
        parent_tree: List = self.parent.get_parent_tree()
        # add self to list
        parent_tree.append(self)
        # filter list for duplicates
        parent_tree = remove_duplicates_from_list(parent_tree)

        return parent_tree

    def export_gcode(self,
                     resolution_obj: ListOfPoints,
                     canvas_xlimit: int = 330,
                     canvas_ylimit: int = 330,
                     canvas_axes_buffer: int = 10,
                     point_array_starting_offset: int = 0,
                     decimal_precision: int = 1,
                     file_name: str = "spiral"):

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
                resolution_obj: ListOfPoints,
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

        fig_static = plt.figure(figsize=[6, 6])
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+700+100")

        # add figure and subplots to makes axes
        fig = plt.figure(figsize=[6, 6])
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+100+100")

        ax_static = fig_static.add_subplot(111, frameon=False)
        ax = fig.add_subplot(111, frameon=False)

        fig_static.tight_layout()
        fig.tight_layout()

        plt.axis('off')

        # get figure limits from drawing objs
        self_xmin, self_xmax, self_ymin, self_ymax = self.get_min_max_values(
            point_array_starting_offset=point_array_starting_offset, buffer=5)

        xmin = self_xmin
        xmax = self_xmax
        ymin = self_ymin
        ymax = self_ymax

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
        # helps with + and - infinity limits
        max_limit = 2000
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

        ax_static.set_xlim(left=self_xmin, right=self_xmax)
        ax_static.set_ylim(bottom=self_ymin, top=self_ymax)
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
            self, foundation_list_of_points: ListOfPoints) -> None:
        raise NotImplementedError

    def get_parent_points(self,
                          foundation_list_of_points: ListOfPoints) -> None:
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

    def create_point_lists(self, foundation_list_of_points: ListOfPoints):
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
                 starting_angle: float = 0,
                 deg: bool = False,
                 parent: Anchorable = Anchor(0 + 0j)):

        super().__init__()
        # circle defined by C * exp( 2*pi*j * freq * resolution_ticker )
        # C = length * exp( starting_angle * j )
        self.rotational_frequency: float = frequency

        self.radius: float = radius

        if not deg:
            self.starting_angle: float = starting_angle
        else:
            self.starting_angle: float = math.radians(starting_angle)

        # constant
        # self.circle_constants: complex = self.radius * np.exp(self.starting_angle * 1j * self.exp_const * self.rotational_frequency)

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

    def create_point_lists(self, resolution: ListOfPoints) -> None:
        if self.point_array is None:
            self.parent.create_point_lists(resolution)
            # parent points + radius * exp(i * angle)
            # angle = starting angle + (rotational multiplier * base rotation)
            self.point_array = self.parent.point_array + (self.radius * np.exp(
                1j * (self.starting_angle +
                      (self.rotational_frequency * resolution.point_array))))

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
                           point_array_starting_offset: int = 0) -> Tuple:
        return super().get_min_max_values(
            buffer=buffer,
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
        return f'radius: {self.radius}, freq: {self.rotational_frequency}, angle: {self.starting_angle}'


class Curve(Anchorable, BarMateSlide):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float = 0,
        deg: bool = False,
    ):

        super().__init__()
        self.parent: Type[Anchorable] = parent

        if not deg:
            self.starting_angle: float = starting_angle
        else:
            # convert from deg to radians for internal math
            self.starting_angle: float = math.radian(starting_angle)

        self.rotational_frequency: float = frequency

        self.color: str = random.choice(color_hex)
        """
        Drawing Objects
        """
        self.drawing_object_spiral = plt.Line2D(
            # [], [], marker='*', markevery=(1, 1), linestyle='--')
            [],
            [],
            linestyle='--',
            color=self.color)
        self.drawing_object_parent2spiral = plt.Line2D(
            # [], [], marker='x', markevery=(1, 1), linestyle='-.')
            [],
            [],
            marker='x',
            markevery=None,
            linestyle='-.',
            color=self.color)

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
                           point_array_starting_offset: int = 0) -> Tuple:
        return super().get_min_max_values(
            buffer=buffer,
            point_array_starting_offset=point_array_starting_offset)

    def get_parent_tree(self) -> List:
        return super().get_parent_tree()


# class RotatingCurve(Curve):
#     def __init__(self,
#                  parent: Type[Anchorable],
#                  starting_angle: float,
#                  frequency: float,
#                  deg: bool = False):
#         super().__init__(parent=parent, starting_angle=starting_angle, deg=deg)
#         self.rotational_frequency = frequency


class ArchimedeanSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        radius_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = radius_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)

        self.point_array = self.parent.point_array + \
            (self.a * (self.rotational_frequency * lop.point_array)) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class HyperbolicSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        radius_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = radius_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)

        self.point_array = self.parent.point_array + \
            (self.a / (self.rotational_frequency * lop.point_array)) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class FermatSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        radius_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = radius_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)

        self.point_array = self.parent.point_array + \
            (self.a * np.sqrt((self.rotational_frequency * lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class LituusSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        radius_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = radius_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)

        self.point_array = self.parent.point_array + \
            (self.a / np.sqrt((self.rotational_frequency * lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class LogarithmicSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        radius_mod: float,
        tangential_angle: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = radius_mod
        self.k = tangential_angle

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)

        self.point_array = self.parent.point_array + \
            (self.a * np.exp(self.k * (self.rotational_frequency * lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class PoinsotBoundedSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        alpha: float,
        kappa: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = alpha
        self.k = kappa

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)

        self.point_array = self.parent.point_array + \
            ((((self.a * np.cos(self.rotational_frequency * lop.point_array)) / (1/np.sinh(self.k * self.rotational_frequency * lop.point_array))) ) + \
                1j * ((self.a * np.sin(self.rotational_frequency * lop.point_array) / (1/np.sinh(self.k * self.rotational_frequency * lop.point_array))))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class PoinsotAsymptoteSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        alpha: float,
        kappa: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = alpha
        self.k = kappa

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)

        self.point_array = self.parent.point_array + \
            ((((self.a * np.cos(self.rotational_frequency * lop.point_array)) / (1/np.cosh(self.k * self.rotational_frequency * lop.point_array))) ) + \
                1j * ((self.a * np.sin(self.rotational_frequency * lop.point_array) / (1/np.cosh(self.k * self.rotational_frequency * lop.point_array))))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class DopplerSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        radius_mod: float,
        doppler_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = radius_mod
        self.k = doppler_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * ((lop.point_array * np.cos(lop.point_array)) + (self.k * lop.point_array) )) + \
            1j*(self.a * lop.point_array * np.sin(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class AbdankAbakanowicz(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        radius_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.r = radius_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
                ((self.r * np.sin(lop.point_array)) + 1j*((np.power(self.r,2)*(lop.point_array + (np.sin(lop.point_array) * np.cos(lop.point_array))))/2)) * \
                np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class WitchOfAgnesi(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        radius_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = radius_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.tan(lop.point_array)) + 1j*(self.a * np.power(np.cos(lop.point_array),2))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Anguinea(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        apex_width: float,
        apex_height: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = apex_width
        self.d = apex_height

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.d * np.tan(lop.point_array / 2)) + 1j * ((self.a/2) * np.sin(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Besace(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_mod: float,
        y_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_mod
        self.b = y_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (((self.a * np.cos(lop.point_array)) - (self.b * np.sin(lop.point_array))) + 1j * (((self.a * np.cos(lop.point_array)) - (self.b * np.sin(lop.point_array))) * -np.sin(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class BicornRegular(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * np.sin(lop.point_array)) + 1j * ((self.a * np.power(np.cos(lop.point_array),2))/(2 + np.cos(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class BicornCardioid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        translate_x: float = 0,
        translate_y: float = 0,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.translate_x = translate_x
        self.translate_y = translate_y

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((((4 *self.a* np.cos(lop.point_array) * (2 - (np.sin(lop.point_array) *  np.power(np.cos(lop.point_array),2))))/(2 + np.sin(lop.point_array) + (2 * np.power(np.cos(lop.point_array),2)))) + self.translate_x) + \
                1j * (((4 * self.a *(1 + np.power(np.cos(lop.point_array),2) + np.power(np.cos(lop.point_array),4)))/(2 + np.sin(lop.point_array) + (2 * np.power(np.cos(lop.point_array),2)))) +  self.translate_y)) * \
                np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Bifolium(Curve):
    # https://mathcurve.com/courbes2d.gb/bifoliumregulier/bifoliumregulier.shtml
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        loop_x_axis_intersection_point: float,
        size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = loop_x_axis_intersection_point
        self.b = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (((self.a + (self.b * lop.point_array))/(np.power(1 + np.power(lop.point_array,2),2))) + 1j * (((self.a + (self.b * lop.point_array))/(np.power(1 + np.power(lop.point_array,2),2))) * lop.point_array)) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class BifoliumRegular(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * np.sin(self.rotational_frequency * lop.point_array) * np.sin(2*(self.rotational_frequency * lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Biquartic(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size_mod: float,
        y_size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.sin(3 * lop.point_array) * np.cos(lop.point_array)) + 1j * (self.b * np.power(np.sin(3 * lop.point_array) * np.sin(lop.point_array),2))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class BoothOvals(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        alpha: float,
        beta: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = alpha
        self.b = beta

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (((self.a * np.power(self.b,2) * np.cos(lop.point_array))/((np.power(self.b,2)*np.power(np.cos(lop.point_array),2)) + (np.power(self.a,2)*np.power(np.sin(lop.point_array),2)))) + \
            1j * ((np.power(self.a,2)*self.b*np.sin(lop.point_array)) / ((np.power(self.b,2)*np.power(np.cos(lop.point_array),2)) + (np.power(self.a,2)*np.power(np.sin(lop.point_array),2))))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class BoothLemniscates(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        alpha: float,
        beta: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = alpha
        self.b = beta

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (((self.b * np.sin(lop.point_array)) / (1 + ((np.power(self.a,2))/(np.power(self.b,2)) * np.power(np.cos(lop.point_array),2)))) + \
            1j *((self.a * np.sin(lop.point_array) * np.cos(lop.point_array)) / (1 + ((np.power(self.a,2))/(np.power(self.b,2)) * np.power(np.cos(lop.point_array),2))))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Kiss(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size_mod: float,
        y_size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.cos(lop.point_array)) + 1j*(self.b * np.power(np.sin(lop.point_array),3))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Clairaut(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        shape_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.n = shape_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * np.power(np.sin(self.rotational_frequency * lop.point_array),self.n)) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class ConchoidOfNicomedes(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        const_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.b = const_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a / np.cos(self.rotational_frequency * lop.point_array)) + self.b) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Cornoid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size_mod: float,
        y_size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.cos(lop.point_array) * np.cos(2*lop.point_array)) + 1j * (self.b * np.sin(lop.point_array) * (2 + np.cos(2*lop.point_array)))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class MalteseCross(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size_mod: float,
        y_size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ( ((self.a * np.cos(lop.point_array)) * (np.power(np.cos(lop.point_array),2) - 2) ) + 1j*(self.b * np.sin(lop.point_array) * np.power(np.cos(lop.point_array),2))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class RationalCircularCubic(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        alpha: float,
        beta: float,
        delta: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = alpha
        self.b = beta
        self.d = delta

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ( ((self.d * np.power(lop.point_array,2)) + (2*self.b*lop.point_array) + (2*self.a) + self.d ) / (1 + np.power(lop.point_array,2)) ) +\
            1j * (lop.point_array * ( ((self.d * np.power(lop.point_array,2)) + (2*self.b*lop.point_array) + (2*self.a) + self.d ) / (1 + np.power(lop.point_array,2)) )) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class SluzeCubic(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        alpha: float,
        beta: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = alpha
        self.b = beta

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a/np.cos(self.rotational_frequency * lop.point_array)) + ((self.b*self.b/self.a) * np.cos(self.rotational_frequency * lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class InvoluteCircle(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * (np.cos(lop.point_array) + (lop.point_array * np.sin(lop.point_array)))) + 1j*(self.a * (np.sin(lop.point_array) - (lop.point_array * np.cos(lop.point_array))))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class DumbBell(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        width: float,
        height: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = height
        self.b = width

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.cos(lop.point_array)) + 1j*((np.power(self.a,2)*np.power(np.cos(lop.point_array),2)*np.sin(lop.point_array))/self.b)) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class StraightLine(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x2: float,
        y2: float,
        y1: float = 0,
        x1: float = 0,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=0,
                         frequency=frequency,
                         deg=deg)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        dx = (self.x2 - self.x1) / math.tau
        dy = (self.y2 - self.y1) / math.tau
        # dy = y2 - y1
        self.point_array = self.parent.point_array + \
            ((self.x1 + (dx * lop.point_array)) + 1j * (self.y1 + (dy * lop.point_array))) * \
            np.exp(1j * (self.rotational_frequency * lop.point_array))
        # ((x1 + (dx * lop.point_array)) + 1j*(y1 + (dy * lop.point_array))) * \


class DurerShellCurve(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        alpha: float,
        beta: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = alpha
        self.b = beta

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((((self.a*np.cos(lop.point_array))/(np.cos(lop.point_array)-np.sin(lop.point_array))) + (self.b*np.cos(lop.point_array))) + 1j*(self.b*np.sin(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Ellipse(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        height: float,
        width: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = width
        self.b = height

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.cos(lop.point_array)) + 1j*(self.b*np.sin(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Rose(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        num_of_arms: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.n = num_of_arms

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * np.cos(self.n * (self.rotational_frequency * lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Folioid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        alpha: float,
        eta: float,
        num_of_arms: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = alpha
        self.e = eta
        self.n = num_of_arms

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * (self.e * np.cos(self.n * (self.rotational_frequency * lop.point_array)) + np.sqrt(1 - (np.power(self.e,2) * np.power(np.sin(self.n * (self.rotational_frequency * lop.point_array)),2))))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class SimpleFolium(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a / np.power(1+np.power(lop.point_array,2),2)) + 1j*((self.a / np.power(1+np.power(lop.point_array,2),2))*lop.point_array)) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Folium(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        radius: float,
        alpha: float,
        beta: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.r = radius
        self.a = alpha
        self.b = beta

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.r * np.cos(3*(self.rotational_frequency * lop.point_array))) + (self.a * np.cos(self.rotational_frequency * lop.point_array)) + (self.b * np.sin(self.rotational_frequency * lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))
        # ((self.r * np.cos(3*(lop.point_array))) + (self.a * np.cos(lop.point_array)) + (self.b * (lop.point_array))) * \


class Cochleoid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        num_of_nodes: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.n = num_of_nodes + 2

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ( (self.a * np.sin((self.n*(self.rotational_frequency * lop.point_array))/(self.n-1)) ) / (self.n*np.sin((self.rotational_frequency * lop.point_array)/(self.n-1))) ) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class ConstantAngularAccelerationSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
        (self.a * np.power(self.rotational_frequency * lop.point_array,2) / 2) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class GalileanSpiral(Curve):  #
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        alpha: float,
        beta: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = alpha
        self.b = beta

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
        (self.a + (self.b * np.power(self.rotational_frequency * lop.point_array,2))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class HourGlass(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.sin(lop.point_array)) + 1j*(self.a * np.sin(lop.point_array) * np.cos(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Aerofoil(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        a: float,
        c: float,
        d: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = a
        self.c = c
        self.d = d

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        r = np.abs(self.c + (1j * self.d) - self.a)
        z0 = self.c + (1j * self.d) + (
            r * np.exp(1j * (self.rotational_frequency * lop.point_array)))
        self.point_array = self.parent.point_array + \
            ((1/2) * (z0 + (np.power(self.a,2)/z0))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Piriform(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size: float,
        y_size: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size
        self.b = y_size

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * (1 + np.sin(lop.point_array))) + 1j*(self.b * np.cos(lop.point_array) * (1 + np.sin(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Teardrop(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        shape_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.n = shape_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.cos(lop.point_array)) + 1j*(self.a * np.sin(lop.point_array) * np.power((np.sin(lop.point_array/2)),self.n))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class FigureEight(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (((self.a * np.sin(lop.point_array))/(1 + np.power(np.cos(lop.point_array),2))) + 1j * ((self.a*np.sin(lop.point_array)*np.cos(lop.point_array))/(1 + np.power(np.cos(lop.point_array),2)))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Lissajous(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size: float,
        y_size: float,
        angle_mod: float,
        angle_shift: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size
        self.b = y_size
        self.n = angle_mod
        self.phi = angle_shift

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.sin(self.rotational_frequency * lop.point_array)) + 1j*(self.b * np.sin((self.n*(self.rotational_frequency * lop.point_array))+self.phi))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Egg(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        radius1: float,
        radius2: float,
        center_offset: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = radius1
        self.b = radius2
        self.d = center_offset

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (((np.sqrt((self.a*self.a) - (self.d*self.d*np.power(np.sin(lop.point_array),2)))+(self.d*np.cos(lop.point_array)))*np.cos(lop.point_array)) + \
            1j*(self.b * np.sin(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class DoubleEgg(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * np.power(np.cos(self.rotational_frequency * lop.point_array),2)) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class HabenichtTrefoil(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        num_of_pedals: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.n = num_of_pedals

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * (1 + np.cos(self.n*(self.rotational_frequency * lop.point_array)) + np.power(np.sin(self.n*(self.rotational_frequency * lop.point_array)),2))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class LaporteHeart(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size: float,
        y_size: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size
        self.b = y_size

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a*(np.power(np.sin(lop.point_array),3))) + 1j*(self.b*(np.cos(lop.point_array) - np.power(np.cos(lop.point_array),4)))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class BossHeart(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size: float,
        y_size: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size
        self.b = y_size

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a*np.cos(lop.point_array)) + 1j*(self.b*(np.sin(lop.point_array) + np.sqrt(np.abs(np.cos(lop.point_array)))))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Propeller(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * np.sqrt(np.sin(4 * (self.rotational_frequency * lop.point_array)) / (1 - ((3/4) * np.power(np.sin(4 * (self.rotational_frequency * lop.point_array)),2))) )) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class TFayButterfly(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        flair: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.b = flair

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a * (np.exp(np.cos(self.rotational_frequency * lop.point_array)) - (2 * np.cos(4 * (self.rotational_frequency * lop.point_array))) + (self.b * np.power(np.sin((self.rotational_frequency * lop.point_array)/5),5)))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Parabola(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        shape_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = shape_mod
        self.b = size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.b * ((lop.point_array) + 1j * (self.a * np.power(lop.point_array,2)))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Fish(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        fishiness: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.k = fishiness

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a*(np.cos(lop.point_array) + 2*self.k*np.cos(lop.point_array/2))) + 1j*(self.a*np.sin(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class DoubleFish(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size_mod: float,
        y_size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * ((5 * np.cos(lop.point_array)) - ((np.sqrt(2)-1)*(np.cos(5* lop.point_array))))) + 1j*(self.b * np.sin(4 * lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Polygasteroid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        eta: float,
        num_of_arms: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.e = eta
        self.n = num_of_arms

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a / (1 + (self.e * np.cos(self.n * self.rotational_frequency * lop.point_array)))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Sinusoid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        size_mod: float,
        height_mod: float,
        wave_freq_mod: float,
        rotational_frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=rotational_frequency,
                         deg=deg)
        self.c = size_mod
        self.a = height_mod
        self.b = 1 / wave_freq_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.c * ((lop.point_array) + 1j * (self.a * np.sin(lop.point_array/self.b)))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class BalanceSpring(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        frequency: float,
        size_mod: float,
        winding_constant: float,
        spring_stiffness: float,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = size_mod
        self.k = winding_constant  # smaller num = more windings
        self.m = spring_stiffness  # larger num = less windings

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            (self.a / (1 + (self.k * np.exp(self.m * self.rotational_frequency * lop.point_array)))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Talbot(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size_mod: float,
        y_size_mod: float,
        c2: float = None,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size_mod
        self.b = y_size_mod
        if c2 is None:
            self.c2 = np.power(self.a, 2) - np.power(self.b, 2)
        else:
            self.c2 = c2
        # self.c2 = c2

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((  self.a * np.cos(lop.point_array) * (1 + (self.c2 * np.power(np.sin(lop.point_array),2) / np.power(self.a,2)) )) + \
            1j*(self.b * np.sin(lop.point_array) * (1 - (self.c2 * np.power(np.cos(lop.point_array),2) / np.power(self.b,2))) )) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class TNConstant(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size_mod: float,
        y_size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * (np.power(np.cos(lop.point_array),2) + np.log(np.sin(lop.point_array)))) + \
            1j*(self.b * np.sin(lop.point_array) * np.cos(lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class CevaTrisectrix(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size_mod: float,
        y_size_mod: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a *(np.cos(3*lop.point_array) + (2*np.cos(lop.point_array)))) + 1j*(self.b * np.sin(3 * lop.point_array))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class Basin2D(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        starting_angle: float,
        x_size_mod: float,
        y_size_mod: float,
        num_of_pedals: float,
        pedal_rotation: float,
        frequency: float = 0,
        deg: bool = False,
    ):
        super().__init__(parent=parent,
                         starting_angle=starting_angle,
                         frequency=frequency,
                         deg=deg)
        self.a = x_size_mod
        self.b = y_size_mod
        self.n = num_of_pedals
        self.phi = pedal_rotation

    def create_point_lists(self, lop: ListOfPoints) -> None:
        self.parent.create_point_lists(lop)
        self.point_array = self.parent.point_array + \
            ((self.a * np.cos(lop.point_array + self.phi) * np.cos(self.n*lop.point_array)) +\
            1j*(self.b * np.power(np.cos(self.n * lop.point_array),2))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))


class FreeBar(Anchorable):
    def __init__(self,
                 parent: Type[Anchorable],
                 link_point_length_from_parent: float,
                 arm_length: float = 0,
                 arm_angle: float = 0,
                 deg: bool = False):
        super().__init__()
        self.parent: Type[Anchorable] = parent
        self.link_point_length = link_point_length_from_parent
        self.arm_length: float = arm_length
        if deg:
            self.arm_angle: float = np.radians(arm_angle)
        else:
            self.arm_angle: float = arm_angle

        self.link_point_array: Union[np.ndarray, None] = None

        self.dotted_color: str = random.choice(color_hex)
        self.solid_color: str = random.choice(color_hex)

        """ Drawing Objects """
        self.parent2mate_dotted_line = plt.Line2D([], [],
                                                  marker='D',
                                                  markevery=(1, 1),
                                                  linestyle='--',
                                                  color=self.dotted_color)

        self.parent2link_solid_line = plt.Line2D([], [],
                                                 marker='x',
                                                 markevery=(1, 1),
                                                 color=self.solid_color)

        self.link2arm_solid_line = plt.Line2D([], [],
                                              marker='.',
                                              markevery=(1, 1),
                                              color=self.solid_color)

    def get_main_drawing_objects(self) -> List:
        return [
            self.parent2mate_dotted_line, self.parent2link_solid_line,
            self.link2arm_solid_line
        ]


class _Point2PointBar(FreeBar):
    def __init__(self,
                 parent: Type[Anchorable],
                 mate: Type[Anchorable],
                 link_point_length_from_parent: float = 0,
                 arm_length: float = 0,
                 arm_angle: float = 0):

        super().__init__(
            parent=parent,
            link_point_length_from_parent=link_point_length_from_parent,
            arm_length=arm_length,
            arm_angle=arm_angle)

        self.mate = mate

    def update_drawing_objects(self, frame) -> None:
        self.parent2mate_dotted_line.set_data([
            self.parent.point_array[frame].real,
            self.mate.point_array[frame].real
        ], [
            self.parent.point_array[frame].imag,
            self.mate.point_array[frame].imag
        ])

        self.parent2link_solid_line.set_data([
            self.parent.point_array[frame].real,
            self.link_point_array[frame].real
        ], [
            self.parent.point_array[frame].imag,
            self.link_point_array[frame].imag
        ])

        self.link2arm_solid_line.set_data(
            [self.link_point_array[frame].real, self.point_array[frame].real],
            [self.link_point_array[frame].imag, self.point_array[frame].imag])

    def create_point_lists(self,
                           foundation_list_of_points: ListOfPoints) -> None:
        if self.point_array is None:
            self.parent.create_point_lists(foundation_list_of_points)
            self.mate.create_point_lists(foundation_list_of_points)

            self.link_point_array = self.parent.point_array + \
                (self.link_point_length * np.exp(1j * np.angle(self.mate.point_array - self.parent.point_array)))

            self.point_array = self.link_point_array + \
                (self.arm_length * np.exp(1j * (np.angle(self.mate.point_array - self.parent.point_array) + self.arm_angle)))

    def get_parent_tree(self) -> List:
        # get obj tree from this obj parent tree
        # which includes itself
        parent_tree: List = super().get_parent_tree()
        # get obj tree for the mate, doesn't this obj
        parent_tree.extend(self.mate.get_parent_tree())
        # eliminate duplicate objs if parent and mate share objs in the trees
        parent_tree = remove_duplicates_from_list(parent_tree)

        return parent_tree


# TODO
# class Point2ShapeSliderBar
# fix point to an equation
# think slider-crank mechanism but not just linear motion
# can it be done with other shapes?
# equations?


class Point2PointSliderBar(_Point2PointBar):
    def __init__(self,
                 parent: Type[Anchorable],
                 mate: Type[Anchorable],
                 link_point_length_from_parent: float,
                 arm_length: float = 0,
                 arm_angle: float = 0):

        super().__init__(
            parent=parent,
            mate=mate,
            link_point_length_from_parent=link_point_length_from_parent,
            arm_length=arm_length,
            arm_angle=arm_angle,
        )


class Point2PointElasticBar(_Point2PointBar):
    def __init__(self,
                 parent: Type[Anchorable],
                 mate: Type[Anchorable],
                 link_point_percentage_distance_from_parent_to_mate: float,
                 arm_length: float = 0,
                 arm_angle: float = 0):

        super().__init__(parent=parent,
                         mate=mate,
                         link_point_length_from_parent=0,
                         arm_length=arm_length,
                         arm_angle=arm_angle)

        self.link_point_percentage = link_point_percentage_distance_from_parent_to_mate / 100

    def create_point_lists(self,
                           foundation_list_of_points: ListOfPoints) -> None:
        if self.point_array is None:
            self.parent.create_point_lists(foundation_list_of_points)
            self.mate.create_point_lists(foundation_list_of_points)

            self.link_point_length = self.link_point_percentage * (np.sqrt(
                np.power(
                    self.mate.point_array[:].real -
                    self.parent.point_array[:].real, 2) + np.power(
                        self.mate.point_array[:].imag -
                        self.parent.point_array[:].imag, 2)))
            super().create_point_lists(foundation_list_of_points)

            # self.link_point_array = self.parent.point_array + \
            #     ((self.link_point_percentage * (self.mate.point_array - self.parent.point_array)) * \
            #     np.exp(1j * np.angle(self.mate.point_array - self.parent.point_array)))

            # self.point_array = self.link_point_array + \
            #     (self.arm_length * np.exp(1j * (np.angle(self.mate.point_array - self.parent.point_array) + self.arm_angle)))


class TwoBarLinkage(Anchorable):
    def __init__(self,
                 primary_linkage: FreeBar,
                 secondary_linkage: FreeBar,):

        super().__init__()
        self.primary_linkage: FreeBar = primary_linkage
        self.secondary_linkage: FreeBar = secondary_linkage
        self.linkage_point_array: Union[np.ndarray, None] = None

    def create_point_lists(self, foundation_list_of_points: ListOfPoints) -> None:
        if self.point_array is None:
            self.primary_linkage.parent.create_point_lists(foundation_list_of_points)
            self.secondary_linkage.parent.create_point_lists(foundation_list_of_points)

            x0 = self.primary_linkage.parent.point_array.real
            y0 = self.primary_linkage.parent.point_array.imag
            r0 = self.primary_linkage.link_point_length

            x1 = self.secondary_linkage.parent.point_array.real
            y1 = self.secondary_linkage.parent.point_array.imag
            r1 = self.secondary_linkage.link_point_length

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

            #! Gross, manually setting primary and secondary linkage point array feels shitty
            self.linkage_point_array = x4 + (1j * y4)
            # self.secondary_linkage.point_array = self.linkage_point_array

            self.point_array = self.linkage_point_array + \
                (self.primary_linkage.arm_length * \
                np.exp(1j * (np.angle(self.linkage_point_array - self.primary_linkage.parent.point_array) + self.primary_linkage.arm_angle)))
            # self.primary_linkage.point_array = self.point_array


            # doing the below calc in fewer varialbes to save on memory
            # a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
            # h = sqrt(r0 ** 2 - a ** 2)
            # x2 = x0 + a * (x1 - x0) / d
            # y2 = y0 + a * (y1 - y0) / d
            # x3 = x2 + h * (y1 - y0) / d
            # y3 = y2 - h * (x1 - x0) / d
    
    def get_parent_tree(self) -> List:
        # * Careful in here, FreeBar objs aren't included in parent tree!
        # get primary linkage parernt tree without itself
        parent_tree: List = self.primary_linkage.parent.get_parent_tree()
        # get secondary linkage parent tree without itself and add it to primary parent tree
        parent_tree.extend(self.secondary_linkage.parent.get_parent_tree())
        # add self (twobarlinkage obj)
        parent_tree.append(self)
        # remove dups from the list
        parent_tree = remove_duplicates_from_list(parent_tree)

        return parent_tree
    
    def get_main_drawing_objects(self) -> List:
        return [
            self.primary_linkage.parent2link_solid_line,
            self.primary_linkage.link2arm_solid_line,
            self.secondary_linkage.parent2link_solid_line
        ]

    def update_drawing_objects(self, frame) -> None:
        self.primary_linkage.parent2link_solid_line.set_data(
            [self.primary_linkage.parent.point_array[frame].real, self.linkage_point_array[frame].real],
            [self.primary_linkage.parent.point_array[frame].imag, self.linkage_point_array[frame].imag]
        )
        self.primary_linkage.link2arm_solid_line.set_data(
            [self.linkage_point_array[frame].real, self.point_array[frame].real],
            [self.linkage_point_array[frame].imag, self.point_array[frame].imag]
        )
        self.secondary_linkage.parent2link_solid_line.set_data(
            [self.secondary_linkage.parent.point_array[frame].real, self.linkage_point_array[frame].real],
            [self.secondary_linkage.parent.point_array[frame].imag, self.linkage_point_array[frame].imag]
        )



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

        self.pre_arm_point_array: Union[np.ndarray, None] = None
        """ Drawing Objects """
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

        # self.secondary_mate_line = plt.Line2D([], [],
        #                                       marker='D',
        #                                       markevery=(1, 1),
        #                                       linestyle='--')
        # self.secondary_pre_arm_point_line = plt.Line2D([], [],
        #                                                marker='x',
        #                                                markevery=(1, 1))
        # self.secondary_arm_line = plt.Line2D([], [],
        #                                      marker='x',
        #                                      markevery=(1, 1))

    def get_parent_points(self,
                          foundation_list_of_points: ListOfPoints) -> None:
        self.parent.create_point_lists(foundation_list_of_points)

    def create_point_lists(self,
                           foundation_list_of_points: ListOfPoints) -> None:
        if self.point_array is None:
            self.get_parent_points(foundation_list_of_points)

            #  mate is a Mate Fix (such as a bar)
            if isinstance(self.mate, BarMateFix):
                self.get_circle_intersection_with_mate(
                    foundation_list_of_points)
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

    def get_circle_intersection_with_mate(
            self, foundation_list_of_points: ListOfPoints) -> None:
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

            self.mate_point_array = x4 + (1j * y4)
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
        # self.secondary_mate_line.set_data([
        #     0, self.mate_point_array[frame].real -
        #     self.parent.point_array[frame].real
        # ], [
        #     0, self.mate_point_array[frame].imag -
        #     self.parent.point_array[frame].imag
        # ])
        # self.secondary_pre_arm_point_line.set_data([
        #     0, self.pre_arm_point_array[frame].real -
        #     self.parent.point_array[frame].real
        # ], [
        #     0, self.pre_arm_point_array[frame].imag -
        #     self.parent.point_array[frame].imag
        # ])
        # self.secondary_arm_line.set_data(
        #     [
        #         self.pre_arm_point_array[frame].real -
        #         self.parent.point_array[frame].real,
        #         self.point_array[frame].real -
        #         self.parent.point_array[frame].real
        #     ],
        #     [
        #         self.pre_arm_point_array[frame].imag -
        #         self.parent.point_array[frame].imag,
        #         self.point_array[frame].imag -
        #         self.parent.point_array[frame].imag
        #     ],
        # )

    def get_main_drawing_objects(self) -> List:
        # return [self.main_pre_arm_point_line, self.main_mate_line, self.main_arm_line, self.mate_intersection_circle]
        return [
            self.main_pre_arm_point_line, self.main_mate_line,
            self.main_arm_line
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

    # def get_secondary_drawing_objects(self) -> List:
    #     return [
    #         self.secondary_pre_arm_point_line, self.secondary_mate_line,
    #         self.secondary_arm_line
    #     ]

    # ? should parent be normalized or another point? MATE?
    # def get_min_max_values_normalized_to_origin(self,
    #                                             buffer: float = 0) -> Tuple:
    #     x_min = min(
    #         itertools.chain(
    #             self.point_array[:].real - self.parent.point_array[:].real,
    #             self.mate_point_array[:].real -
    #             self.parent.point_array[:].real,
    #             self.pre_arm_point_array[:].real -
    #             self.parent.point_array[:].real, [0])) - buffer
    #     x_max = max(
    #         itertools.chain(
    #             self.point_array[:].real - self.parent.point_array[:].real,
    #             self.mate_point_array[:].real -
    #             self.parent.point_array[:].real,
    #             self.pre_arm_point_array[:].real -
    #             self.parent.point_array[:].real, [0])) + buffer

    #     y_min = min(
    #         itertools.chain(
    #             self.point_array[:].imag - self.parent.point_array[:].imag,
    #             self.mate_point_array[:].imag -
    #             self.parent.point_array[:].imag,
    #             self.pre_arm_point_array[:].imag -
    #             self.parent.point_array[:].imag, [0])) - buffer
    #     y_max = max(
    #         itertools.chain(
    #             self.point_array[:].imag - self.parent.point_array[:].imag,
    #             self.mate_point_array[:].imag -
    #             self.parent.point_array[:].imag,
    #             self.pre_arm_point_array[:].imag -
    #             self.parent.point_array[:].imag, [0])) + buffer

    #     return x_min, x_max, y_min, y_max


a1 = Anchor(-500 + 0j)
a2 = Anchor(500 + 0j)

c1 = Circle(300, 1 / 2, parent=a1)
c2 = Circle(200, 1.35, parent=a2)

# b1 = Bar(c1, 300, c2, 150, arm_length=100, arm_angle=math.tau/4)
# b1 = Bar(c1, 500, c2, arm_length=450)
# b1 = Point2PointSliderBar(c1, c2, 450, 100, math.tau/8)
# b1 = Point2PointElasticBar(c1, c2, 50, 200, math.tau / 8)
b1 = TwoBarLinkage(FreeBar(c1, 1110, 100, math.tau/8),
                   FreeBar(c2, 1100))
# b2 = Bar(c2, 600, c1, mate_object=b1)
# b1.mate = b2

foundation_point_array = ListOfPoints(steps_per_cycle=1440,
                                      cycle_start=0,
                                      cycle_end=10,
                                      num_of_cycles=1)
foundation_point_array.calculate_common_num_of_points()
foundation_point_array.calculcate_points_array()

b1.animate(foundation_point_array, speed=10)
# print (foundation_point_array.point_array)
# s1.animate(foundation_point_array, speed=10)

# s1 = ArchimedeanSpiral(a1, 0, 10, 50)
# s1 = PoinsotAsymptoteSpiral(a1, 0, 1, 10, 1)
# s1 = DopplerSpiral(a1, 0, 1, 0.5, frequency=0.1)
# s1 = AbdankAbakanowicz(a1, math.tau/8, 1)
# s1 = WitchOfAgnesi(a1, 0, 100)
# s1 = Anguinea(a1, 0, 100, 200)
# s1 = Besace(a1, 0, 100, 80)
# s1 = BicornRegular(a1, 0, 100)
# s1 = BicornCardioid(a1, 0, 100, translate_x=-200)
# s1 = Bifolium(a1, 0, 25, 100)
# s1 = BifoliumRegular(a1, 0, 1, 100)
# s1 = Biquartic(a1, 0, 100, 50)
# s1 = BoothOvals(a1, 0, 100, 50)
# s1 = BoothLemniscates(a1, 0, 100, 50)
# s1 = Kiss(a1, 0, 100, 200)
# s1 = Clairaut(a1, 0, 1, 100, 3)
# s1 = Cornoid(a1, 0, 100, frequency=0.55)
# s1 = RationalCircularCubic(a1, 0, 200, 20, 10)
# s1 = SluzeCubic(a1, 0, 1, 100, 400)
# s1 = DumbBell(a1, 0,  100, 200)
# s1 = StraightLine(a1, 200, 500)
# s1 = DurerShellCurve(a1, 0, 200, 500)
# s1 = Ellipse(a1, 0, 200, 500)
# s1 = Rose(a1, 0, 1, 200, 5)
# s1 = Folioid(a1, 0, 1, 200, 1.1, 1/5)
# s1 = SimpleFolium(a1, 0, 200,)
# s1 = Folium(a1, 0, 1, 200, 200, 40)
# s1 = Cochleoid(a1, 0, 1, 200, 2)
# s1 = ConstantAngularAccelerationSpiral(a1, 0, 2, 2)
# s1 = GalileanSpiral(a1, 0, 1, 200, 2)
# s1 = HourGlass(a1, 0, 100)
# s1 = Aerofoil(a1, 0, -20, 5, -10)
# s1 = Piriform(a1, 0, 20, 20)
# s1 = Teardrop(a1, 0, 20, 20)
# s1 = FigureEight(a1, 0, 20)
# s1 = Lissajous(a1, 0, 80, 100, 2/3, math.tau/11)
# s1 = Egg(a1, 0, 100, 60, 80)
# s1 = DoubleEgg(a1, 0, 1, 100)
# s1 = HabenichtTrefoil(a1, 0, 2, 100, 3)
# s1 = LaporteHeart(a1, 0, 100, 50)
# s1 = BossHeart(a1, 0, 100, 10)
# s1 = Propeller(a1, 0, 2, 100)
# s1 = TFayButterfly(a1, 0, 1, 100, 1)
# s1 = Parabola(a1, 0, 10, 0.01)
# s1 = Fish(a1, 0, 50, 1.8, frequency=1.15)
# s1 = DoubleFish(a1, 0, 100, 150, frequency=1.1)
# s1 = Polygasteroid(a1, 0, 2, 100, 0.7, 23/7)
# s1 = Polygasteroid(a1, 0, 2, 100, 0.7, 23/7)
# s1 = Sinusoid(a1, 0, 10, 10, 10)
# s1 = BalanceSpring(a1, 0, 1, 1000, 0.05, 1.1)
# s1 = Talbot(a1, 0, 50, 20, frequency=1.1)
# s1 = TNConstant(a1, 0, 1000, 2000)
# s1 = CevaTrisectrix(a1, 0, 20, 50)
# s1 = Basin2D(a1, 0, 40, 50, 8, math.tau/3)

# b1.animate(base_points, speed=10)
# s1.export_gcode(base_points, point_array_starting_offset=0, decimal_precision=1, file_name="spirangle", canvas_xlimit=205, canvas_ylimit=270)
# s1.animate(base_points, speed=100, point_array_starting_offset=1000)
# print(foundation_point_array.point_array)

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
# ? anamorphosis projections (https://mathcurve.com/courbes2d.gb/anamorphose/anamorphose.shtml)
# ? asteroid https://mathcurve.com/courbes2d.gb/astroid/astroid.shtml
# ? bar draws on a point 1/2 (or some ratio) between parent and mate
# ? bar mate is anywhere on a line x away from parent see: https://mathcurve.com/courbes2d.gb/bernoulli/bernoulli.shtml slider-crank
# ?     https://mathcurve.com/courbes2d.gb/bielledeberard/bielledeberard.shtml
# ? Brocard transformation https://mathcurve.com/courbes2d.gb/brocard/brocard.shtml
# ? https://mathcurve.com/courbes2d.gb/caustic/caustic.htm
# ? https://mathcurve.com/courbes2d.gb/cornu/cornu.shtml
# ? https://mathcurve.com/courbes2d.gb/galilee/galilee.shtml
# ? https://mathcurve.com/courbes2d.gb/gauss/gauss.shtml
# ? https://mathcurve.com/courbes2d.gb/giration/motifs.shtml
# ? https://mathcurve.com/courbes2d.gb/giration/rayonsinusoidal.shtml
# ? https://mathcurve.com/courbes2d.gb/holditch/holditch.shtml
# ? https://mathcurve.com/courbes2d.gb/dentelee/dentelee.shtml
# ? https://mathcurve.com/courbes2d.gb/inverse/inverse.shtml
# ? https://mathcurve.com/courbes2d.gb/isochron/isochrone_paracentrique.shtml
# ? https://mathcurve.com/courbes2d.gb/kieroide/kieroide.shtml
# ? https://mathcurve.com/courbes2d.gb/linteaire/linteaire.shtml
# ? https://mathcurve.com/courbes2d.gb/mascotte/mascotte.shtml
# ? https://mathcurve.com/courbes2d.gb/nageur/nageur.shtml
# ? more hearts: http://www.takayaiwamoto.com/draw_heart/draw_heart.html
# ? https://mathcurve.com/courbes2d.gb/parabolesemicubic/parabolesemicubic.shtml
# ? https://mathcurve.com/courbes2d.gb/piriforme/piriforme.shtml
# ? https://mathcurve.com/courbes2d.gb/poursuite/poursuitemutuelle.shtml
# ? https://mathcurve.com/courbes2d.gb/quarticbicirculairerationnelle/quarticbicirculairerationnelle.shtml
# ? https://mathcurve.com/courbes2d.gb/ramphoid/ramphoide.shtml
# ? https://mathcurve.com/courbes2d.gb/septic/septic.shtml
# ? https://mathcurve.com/courbes2d.gb/sici/sici.shtml
# ? https://mathcurve.com/courbes2d.gb/danseur/danseur.shtml
# ? https://mathcurve.com/courbes2d.gb/tetracuspide/tetracuspide.shtml
# ? https://mathcurve.com/courbes2d.gb/spiraletractrice/spiraletractrice.shtml
# ? https://mathcurve.com/courbes2d.gb/sturm/spirale_sturm.shtml
# ? https://mathcurve.com/courbes2d.gb/sturm/norwichinverse.shtml
# ? https://mathcurve.com/courbes2d.gb/tetracuspide/tetracuspide.shtml
# ? https://mathcurve.com/courbes2d.gb/tractrice/tractrice.shtml
# ? https://mathcurve.com/courbes2d.gb/troisbarres/troisbarre.shtml
# ? https://mathcurve.com/surfaces/surfaces.shtml
# ? https://mathcurve.com/courbes2d.gb/oeuf/oeufgranville.shtml
# ? https://mathcurve.com/courbes2d.gb/trefle/trefle.shtml
# ?
# ?
# see: https://en.wikipedia.org/wiki/Spiral
#      https://en.wikipedia.org/wiki/List_of_spirals
#      https://mathcurve.com/courbes2d.gb/courbes2d.shtml
# ?
# ?
# ?
"""        
=== === === ===

        elif self.type == "hypercissoid":
        self.point_array = self.parent.point_array + \
            () + 1j*() * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))

        elif self.type == "cassinian":
            a = self.spiral_parameters[0]
            b = self.spiral_parameters[1]
            n = self.spiral_parameters[2]
            e = b / a
            # self.point_array = self.parent.point_array + \
            #     (a * np.sqrt(np.cos(2*lop.point_array) +np.sqrt(np.power(e,4) - np.power(np.sin(lop.point_array * 2),2)) )) * np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))
            self.point_array = self.parent.point_array + \
                (a * np.sqrt(np.cos(n*lop.point_array) +np.sqrt(np.power(e,2*n) - np.power(np.sin(lop.point_array * n),2)) )) * \
                np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))

        elif self.type == "alain's-curve":
            if self.spiral_parameters[0] > self.spiral_parameters[1]:
                raise SpiralParametersError("Parameter-0 cannot be larger than parameter-1")
            elif self.spiral_parameters[0] < 0:
                raise SpiralParametersError("Parameter-0 cannot be less than 0")
            a = self.spiral_parameters[0]
            b = self.spiral_parameters[1]
            a2 = np.power(a,2)
            b2 = np.power(b,2)
            c = np.sqrt((a2 + b2)/2)
            k = (b2 - a2)/(a2 + b2)
            self.point_array = self.parent.point_array + \
                ((c * np.sqrt(np.cos(list_of_points.point_array * 2) - k)) / np.cos(list_of_points.point_array * 2)) * \
                np.exp(1j * (self.starting_angle + (self.rotational_frequency * list_of_points.point_array)))

        # elif self.type == "doppler+rotation":
        #     self.point_array = self.parent.point_array + \
        #         np.sqrt(
        #             np.power(self.spiral_parameters[0] * (lop.point_array * np.cos(lop.point_array) + (self.spiral_parameters[1] * lop.point_array)), 2) +
        #             np.power(self.spiral_parameters[0] * lop.point_array * np.sin(lop.point_array), 2)) * \
        #             np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))

        elif self.type == "catenary":
            a = self.spiral_parameters[0]
            k = self.spiral_parameters[1]
            self.point_array = self.parent.point_array + \
            ((a * (lop.point_array + (k * np.sinh(lop.point_array)))) + 1j * (a * ((np.cosh(lop.point_array))+((k/2) * np.power(np.sinh(lop.point_array),2))))) * \
            np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))
            # ((a * np.log(lop.point_array)) + 1j * ((a/2) * (lop.point_array + (1/lop.point_array)))) * np.exp(1j * self.starting_angle)

        elif self.type == "devil":
            a = self.spiral_parameters[0]
            b = self.spiral_parameters[1]
            self.point_array = self.parent.point_array + \
                (np.sqrt((b*b) + ((a*a)/np.cos(2*lop.point_array)))) * \
                np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))

        elif self.type == "parafolium":
            a = self.spiral_parameters[0]
            b = self.spiral_parameters[1]
            self.point_array = self.parent.point_array + \
                (((a * np.cos(2*lop.point_array)) + ((b/2)*np.sin(2*lop.point_array)))/np.power(np.cos(lop.point_array),3) ) * \
                np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))
        
        elif self.type == "elastic":  # needs work
            a = self.spiral_parameters[0]
            k = self.spiral_parameters[1]
            # ( (a * np.sqrt(k + np.cos(lop.point_array))) + 1j * ((a/2) * (np.cos(lop.point_array)/(k + np.cos(lop.point_array)))) ) * \
            self.point_array = self.parent.point_array + \
                ( (a * np.sqrt(k + np.cos(lop.point_array))) + 1j * ((2*(a * np.sqrt(k + np.cos(lop.point_array))))/np.power(a,2))) * \
                np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))

        elif variety == "boddorf":  #meh
                self.point_array = self.parent.point_array + \
                    (np.power(np.abs(np.tan(lop.point_array)),(1/np.sin(lop.point_array)))) * \
                    np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))

        elif variety == "daniel":  #meh
            self.point_array = self.parent.point_array + \
                ((np.sqrt(np.power(np.cos(2*lop.point_array),3) / np.power(np.cos(lop.point_array),2))) + \
                1j*((2*np.sin(2*lop.point_array)) - np.power(np.tan(lop.point_array),2))) * \
                np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))
        
        elif self.type == "trifolium":
            a = self.spiral_parameters[0]
            b = self.spiral_parameters[1]
            r = self.spiral_parameters[2]
            self.point_array = self.parent.point_array + \
                ( (4 * r * np.power(np.cos(lop.point_array),3)) + ((a - (3*r))*np.cos(lop.point_array)) + (b * np.sin(lop.point_array)) ) * \
                np.exp(1j * (self.starting_angle + (self.rotational_frequency * lop.point_array)))
"""
