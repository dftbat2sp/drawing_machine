# import cmath
import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Type, Union, Dict
import random
# from matplotlib import lines
from multipledispatch import dispatch
# import time

import matplotlib.animation as animation
# import matplotlib.gridspec as gridspec
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


class FreeBarHelperClassError(Exception):
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
np.set_printoptions(threshold=5)


def remove_duplicates_from_list(my_list):
    return list(dict.fromkeys(my_list))


class SinglePointList(int):
    def __new__(cls, value):
        return super().__new__(cls, value)

    def __init__(self, value):
        self.num = value

    def __getitem__(self, index):
        return self.num


class ListOfPoints:
    """
    ListOfPoints
    Create a list of numbers
    ---

    """

    num_of_points_max = 0
    lowest_common_multiple_of_num_of_cycles = 1
    final_number_of_points = 0
    list_of_LOP_objects: List = []

    def __init__(
            self,
            steps_per_cycle: int = 1000,
            #  total_num_of_points: int = None,
            #  num_of_rotations: float = 50,
            #  num_of_points_per_rotation: int = None,
            cycle_start: float = 0,
            cycle_end: float = 1,
            num_of_cycles: int = 1,
            treat_as_length_or_radius=False):

        self.list: Union[np.ndarray, None] = None

        if not isinstance(steps_per_cycle, int):
            raise ListOfPointsParameterError('steps_per_cycle is not an INT')
        if not isinstance(num_of_cycles, int):
            raise ListOfPointsParameterError('num_of_cycles is not an INT')
        if treat_as_length_or_radius is False:
            self.mul_by_tau = 1
        else:
            self.mul_by_tau = 0

        self.num_of_cycles: int = num_of_cycles
        self.cycle_start: float = cycle_start
        self.cycle_end: float = cycle_end

        self.cycle_length: float = np.abs(self.cycle_end - self.cycle_start)

        ListOfPoints.list_of_LOP_objects.append(self)
        # if self.cycle_length == 0:
        #     raise ListOfPointsParameterError(
        #         'cycle_end - cycle_start resulted in zero')

        self.initial_num_of_points: int = self.get_num_of_points(
            self.cycle_length, steps_per_cycle, self.num_of_cycles)

        ListOfPoints.num_of_points_max = max(ListOfPoints.num_of_points_max,
                                             self.initial_num_of_points)

        ListOfPoints.lowest_common_multiple_of_num_of_cycles = np.lcm(
            ListOfPoints.lowest_common_multiple_of_num_of_cycles,
            self.num_of_cycles)

    @staticmethod
    def get_num_of_points(cycle_length: float, steps_per_cycle: int,
                          num_of_cycles: int) -> int:
        # cycles = TAU * total_cycles
        # step_size = TAU/step_size
        # typically for #ofPoints you wants total / step size
        # which would be TAU * cycles / TAU / step size == TAU * cycles * step_size / TAU == cycles * step_size
        # +1 to account for fence post problem | - - - - | 1 cycle from 0-1 with 5 step size
        # 0, t/5, 2t/5, 3t/5, 4t/5, 5t/5 <--- 6 points with 1 cycles 5 step size
        # 2 cycles from 0-1 (0-1-0), 5 step size
        # 0, t/5, 2t/5, 3t/5, 4t/5, 5t/5, 4t/5, 3t/5, 2t/5, t/5, 0 = 11 points = (2 cycles * cycle_length * step size) + 1
        # +1 added in calc common num of points
        return math.ceil(cycle_length * steps_per_cycle * num_of_cycles)

    @staticmethod
    def generate_points_lists():
        ListOfPoints.calculate_common_num_of_points()
        for lop in ListOfPoints.list_of_LOP_objects:
            lop.calculcate_points_array()

    @staticmethod
    def calculate_common_num_of_points():
        points_mod_lcm = ListOfPoints.num_of_points_max % ListOfPoints.lowest_common_multiple_of_num_of_cycles
        lcm_minus_points_mod_lcm = 0

        # no need to add anything to num of points if points mod lcm is 0
        if not points_mod_lcm == 0:
            lcm_minus_points_mod_lcm = ListOfPoints.lowest_common_multiple_of_num_of_cycles - points_mod_lcm
        # * still need to add +1
        ListOfPoints.final_number_of_points = ListOfPoints.num_of_points_max + lcm_minus_points_mod_lcm
        # return ListOfPoints.final_number_of_points

    def calculcate_points_array(self) -> None:

        cycle_start_tau: float = self.cycle_start * np.power(
            math.tau, self.mul_by_tau)
        cycle_end_tau: float = self.cycle_end * np.power(
            math.tau, self.mul_by_tau)

        points_per_cycle: int = int(ListOfPoints.final_number_of_points /
                                    self.num_of_cycles)

        print(
            f'points_per_cycle: {points_per_cycle}        num of cycles: {self.num_of_cycles}'
        )
        # print(f'')

        self.list: np.ndarray = np.linspace(cycle_start_tau, cycle_end_tau,
                                            points_per_cycle + 1)

        for cycle_counter in range(self.num_of_cycles - 1):
            # reverse start and end at the beginning of each for loop
            cycle_start_tau, cycle_end_tau = cycle_end_tau, cycle_start_tau
            temp_cycle_point_array = np.linspace(cycle_start_tau,
                                                 cycle_end_tau,
                                                 points_per_cycle + 1)
            # add temp point array minus the first item in the array
            # # ignore the first item because it was included in the first/previous linspace calc
            # np.append(self.list, temp_cycle_point_array[1:])
            self.list = np.append(self.list, temp_cycle_point_array[1:])

            # self.point_array.append(temp_cycle_point_array[1:])

        # self.point_array = np.linspace(self.cycle_start * math.tau,
        #                                self.cycle_end * math.tau,
        #                                ListOfPoints.final_number_of_points)
    # def sqrt(self):
    #     return np.sqrt(self.list)

    def sin(self):
        return np.sin(self.list)

    def cos(self):
        return np.cos(self.list)

    # def power(self, other):
    #     return np.power(self.list, other)

    def __float__(self):
        return self.list

    def __getitem__(self, index):
        return self.list[index]

    @dispatch((int, float, complex, np.ndarray))
    def __add__(self, other):
        return self.list + other

    @dispatch(object)
    def __add__(self, other):
        return self.list + other.list

    @dispatch((int, float, complex, np.ndarray))
    def __sub__(self, other):
        return self.list - other

    @dispatch(object)
    def __sub__(self, other):
        return self.list - other.list

    @dispatch((int, float, complex, np.ndarray))
    def __mul__(self, other):
        return self.list * other

    @dispatch(object)
    def __mul__(self, other):
        return self.list * other.list

    @dispatch((int, float, complex, np.ndarray))
    def __truediv__(self, other):
        return self.list / other

    @dispatch(object)
    def __truediv__(self, other):
        return self.list / other.list

    @dispatch((int, float, complex, np.ndarray))
    def __pow__(self, other):
        return np.power(self.list, other)

    @dispatch(object)
    def __pow__(self, other):
        return np.power(self.list, other.list)

    @dispatch((int, float, complex, np.ndarray))
    def __mod__(self, other):
        return self.list % other

    @dispatch(object)
    def __mod__(self, other):
        return self.list % other.list

    """right side"""

    @dispatch((int, float, complex, np.ndarray))
    def __radd__(self, other):
        return other + self.list

    @dispatch(object)
    def __radd__(self, other):
        return other.list + self.list

    @dispatch((int, float, complex, np.ndarray))
    def __rsub__(self, other):
        return other - self.list

    @dispatch(object)
    def __rsub__(self, other):
        return other.list - self.list

    @dispatch((int, float, complex, np.ndarray))
    def __rmul__(self, other):
        return other * self.list

    @dispatch(object)
    def __rmul__(self, other):
        return other.list * self.list

    @dispatch((int, float, complex, np.ndarray))
    def __rtruediv__(self, other):
        return other / self.list

    @dispatch(object)
    def __rtruediv__(self, other):
        return other.list / self.list

    @dispatch((int, float, complex, np.ndarray))
    def __rpow__(self, other):
        return np.power(other, self.list)

    @dispatch(object)
    def __rpow__(self, other):
        return np.power(other.list, self.list)

    @dispatch((int, float, complex, np.ndarray))
    def __rmod__(self, other):
        return other % self.list

    @dispatch(object)
    def __rmod__(self, other):
        return other.list % self.list


class Anchorable:
    def __init__(self,
                 steps_per_cycle: int = 1000,
                 cycle_start: float = 0,
                 cycle_end: float = 1,
                 num_of_cycles: int = 1):
        self.points: Union[np.ndarray, None] = None
        self.base_points = ListOfPoints(steps_per_cycle=steps_per_cycle,
                                        cycle_start=cycle_start,
                                        cycle_end=cycle_end,
                                        num_of_cycles=num_of_cycles)

    def create_point_lists(self) -> None:
        raise NotImplementedError
        # self.base_points.calculate_common_num_of_points()
        # self.base_points.calculcate_points_array()

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
        x_min = min(self.points[point_array_starting_offset:].real) - buffer
        x_max = max(self.points[point_array_starting_offset:].real) + buffer

        y_min = min(self.points[point_array_starting_offset:].imag) - buffer
        y_max = max(self.points[point_array_starting_offset:].imag) + buffer

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
                    #  resolution_obj: ListOfPoints,
                     canvas_xlimit: int = 330,
                     canvas_ylimit: int = 330,
                     canvas_axes_buffer: int = 10,
                     point_array_starting_offset: int = 0,
                     decimal_precision: int = 2,
                     file_name: str = "spiral"):
        
        ListOfPoints.generate_points_lists()
        self.create_point_lists()

        # TODO simplify straight lines
        # TODO create curves from line segments

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
                if f'{coordinate:.{decimal_precision}f}' != f'{previous_coordinate:.{decimal_precision}f}':
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
        scaled_point_array = self.points[point_array_starting_offset:]

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
                speed: int = 1,
                point_array_starting_offset: int = 0) -> None:

        # get full parent tree of drawer (list of obj)
        obj_needed_for_drawing = self.get_parent_tree()

        # get our own points list
        # self.create_point_lists(resolution_obj)
        ListOfPoints.generate_points_lists()

        speed = int(
            (speed / 100) * (ListOfPoints.final_number_of_points / 1000))

        self.create_point_lists()

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

        ax_static.axis('off')
        ax.axis('off')

        fig_static.tight_layout()
        fig.tight_layout()

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
        max_limit = 5500
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

        line, = ax.plot([], [], linewidth=0.7)

        ax_static.plot(self.points[point_array_starting_offset:].real,
                       self.points[point_array_starting_offset:].imag,
                       linewidth=0.5)

        def on_q(event):
            if event.key == 'q':
                exit()

        def init():
            for artist in artist_list:
                ax.add_artist(artist)

            line.set_data([], [])
            return itertools.chain([line], artist_list)

        def get_frames():
            for i in range(self.points.size - point_array_starting_offset):
                point = i * speed
                if point < self.points.size - point_array_starting_offset:
                    yield point

        def matplotlib_animate(i):

            for obj in obj_needed_for_drawing:
                obj.update_drawing_objects(i + point_array_starting_offset)

            line.set_data(
                self.points[
                    point_array_starting_offset:point_array_starting_offset +
                    i + 1].real,
                self.points[
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


class Anchor(Anchorable):
    def __init__(self, complex_number: complex = 0 + 0j):
        super().__init__(num_of_cycles=1, steps_per_cycle=1)
        self.point = complex_number
        """
        Drawing Objects
        """
        self.main_marker = plt.Line2D([self.point.real], [self.point.imag],
                                      marker="^",
                                      color="#FFAA00")

    def create_point_lists(self):
        #super().create_point_lists()
        self.points = np.full_like(self.base_points.list,
                                   self.point,
                                   dtype=complex)
        # self.point_array = self.point

    def update_drawing_objects(self, frame) -> None:
        pass  # do nothing, point stays still

    def get_main_drawing_objects(self) -> List:
        return [
            self.main_marker,
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


class Circle(Anchorable):
    # exp_const = math.tau * 1j

    def __init__(
            self,
            parent: Anchorable,
            radius: ListOfPoints,
            #  frequency: float,
            starting_angle: float = 0,
            deg: bool = False,
            steps_per_cycle: int = 1000,
            cycle_start: float = 0,
            cycle_end: float = 1,
            num_of_cycles: int = 1):

        super().__init__(steps_per_cycle=steps_per_cycle,
                         cycle_start=cycle_start,
                         cycle_end=cycle_end,
                         num_of_cycles=num_of_cycles)
        # self.rotational_frequency: float = frequency

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
                                                  0,
                                                  fill=False,
                                                  edgecolor=self.color)
        self.main_centre2point_line_artist = plt.Line2D([], [],
                                                        marker='.',
                                                        markevery=(1, 1),
                                                        color=self.color)

    def create_point_lists(self) -> None:
        if self.points is None:
            #super().create_point_lists()
            self.parent.create_point_lists()

            # parent points + radius * exp(i * angle)
            # angle = starting angle + (rotational multiplier * base rotation)
            self.points = self.parent.points + \
                            (self.radius * \
                                np.exp(1j * (self.starting_angle + (self.base_points))))
            # 1j * (self.starting_angle + (self.rotational_frequency * self.base_points.list))))

    def update_drawing_objects(self, frame) -> None:
        # MAIN
        self.main_circle_edge_artist.set_center(
            (self.parent.points[frame].real, self.parent.points[frame].imag))
        self.main_circle_edge_artist.set_radius(self.radius[frame])
        self.main_centre2point_line_artist.set_data(
            [self.parent.points[frame].real, self.points[frame].real],  # x
            [self.parent.points[frame].imag, self.points[frame].imag])  # y

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


class Curve(Anchorable):
    def __init__(self,
                 parent: Type[Anchorable],
                 starting_angle: float = 0,
                 frequency: float = 0,
                 deg: bool = False,
                 steps_per_cycle: int = 1000,
                 cycle_start: float = 0,
                 cycle_end: float = 10,
                 num_of_cycles: int = 1):

        super().__init__(steps_per_cycle=steps_per_cycle,
                         cycle_start=cycle_start,
                         cycle_end=cycle_end,
                         num_of_cycles=num_of_cycles)

        self.parent: Type[Anchorable] = parent

        if not deg:
            self.starting_angle: float = starting_angle
        else:
            # convert from deg to radians for internal math
            self.starting_angle: float = math.radian(starting_angle)

        self.freq_mod: float = frequency

        self.color: str = random.choice(color_hex)
        """Drawing Objects"""
        self.drawing_object_spiral = plt.Line2D(
            # [], [], marker='*', markevery=(1, 1), linestyle='--')
            [],
            [],
            linestyle='solid',
            color=self.color)
        self.drawing_object_parent2spiral = plt.Line2D(
            # [], [], marker='x', markevery=(1, 1), linestyle='-.')
            [],
            [],
            marker='',
            markevery=None,
            linestyle='--',
            color=self.color)

    def update_drawing_objects(self, frame) -> None:
        # spiral drawing obj
        self.drawing_object_spiral.set_data(self.points[:frame].real,
                                            self.points[:frame].imag)

        # center to edge of spriral obj
        self.drawing_object_parent2spiral.set_data(
            [self.parent.points[frame].real, self.points[frame].real],
            [self.parent.points[frame].imag, self.points[frame].imag])

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


class ArchimedeanSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        # frequency: float,
        radius_mod: float,
        deg: bool = False,
        starting_angle: float = 0,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=frequency,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = radius_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * (self.base_points.list)) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))
        # (self.a * (self.rotational_frequency * self.base_points.list)) * \
        # np.exp(1j * (self.starting_angle + (self.rotational_frequency * self.base_points.list)))


class HyperbolicSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        # frequency: float,
        radius_mod: float,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=frequency,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles)
        self.a = radius_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a / (self.base_points.list)) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class FermatSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        # frequency: float,
        radius_mod: float,
        deg: bool = False,
        starting_angle: float = 0,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=frequency,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles)
        self.a = radius_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * np.sqrt((self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class LituusSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        # frequency: float,
        radius_mod: float,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=frequency,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = radius_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a / np.sqrt((self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class LogarithmicSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        radius_mod: float,
        tangential_angle: float,
        # freq_mod: float = 1,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = radius_mod
        self.k = tangential_angle

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * np.exp(self.k * (self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class PoinsotBoundedSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        alpha: float,
        kappa: float,
        freq_mod: float = 1,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = alpha
        self.k = kappa

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((((self.a * np.cos(self.base_points.list)) / (1/np.sinh(self.k * self.base_points.list))) ) + \
                1j * ((self.a * np.sin(self.base_points.list) / (1/np.sinh(self.k * self.base_points.list))))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class PoinsotAsymptoteSpiral(Curve):
    def __init__(self,
                 parent: Type[Anchorable],
                 alpha: float,
                 kappa: float,
                 freq_mod: float = 1,
                 starting_angle: float = 0,
                 deg: bool = False,
                 steps_per_cycle: int = 1000,
                 cycle_start: float = 0,
                 cycle_end: float = 10,
                 num_of_cycles: int = 1):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = alpha
        self.k = kappa

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((((self.a * np.cos(self.base_points.list)) / (1/np.cosh(self.k * self.base_points.list))) ) + \
                1j * ((self.a * np.sin(self.base_points.list) / (1/np.cosh(self.k * self.base_points.list))))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class DopplerSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        radius_mod: float,
        doppler_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = radius_mod
        self.k = doppler_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * ((self.base_points.list * np.cos(self.base_points.list)) + (self.k * self.base_points.list) )) + \
                1j*(self.a * self.base_points.list * np.sin(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class AbdankAbakanowicz(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        radius_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.r = radius_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
                ((self.r * np.sin(self.base_points.list)) + \
                    1j*((np.power(self.r,2)*(self.base_points.list + (np.sin(self.base_points.list) * np.cos(self.base_points.list))))/2)) * \
                np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class WitchOfAgnesi(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        radius_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = radius_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.tan(self.base_points.list)) + \
                1j*(self.a * np.power(np.cos(self.base_points.list),2))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Anguinea(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        apex_width: float,
        apex_height: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = apex_width
        self.d = apex_height

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.d * np.tan(self.base_points.list / 2)) + \
                1j * ((self.a/2) * np.sin(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Besace(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_mod: float,
        y_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_mod
        self.b = y_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (((self.a * np.cos(self.base_points.list)) - (self.b * np.sin(self.base_points.list))) + \
                1j * (((self.a * np.cos(self.base_points.list)) - (self.b * np.sin(self.base_points.list))) * -np.sin(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class BicornRegular(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * np.sin(self.base_points.list)) + \
                1j * ((self.a * np.power(np.cos(self.base_points.list),2))/(2 + np.cos(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class BicornCardioid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        translate_x: float = 0,
        translate_y: float = 0,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.translate_x = translate_x
        self.translate_y = translate_y

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((((4 *self.a* np.cos(self.base_points.list) * (2 - (np.sin(self.base_points.list) *  np.power(np.cos(self.base_points.list),2))))/(2 + np.sin(self.base_points.list) + (2 * np.power(np.cos(self.base_points.list),2)))) + self.translate_x) + \
                1j * (((4 * self.a *(1 + np.power(np.cos(self.base_points.list),2) + np.power(np.cos(self.base_points.list),4)))/(2 + np.sin(self.base_points.list) + (2 * np.power(np.cos(self.base_points.list),2)))) +  self.translate_y)) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Bifolium(Curve):
    # https://mathcurve.com/courbes2d.gb/bifoliumregulier/bifoliumregulier.shtml
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        loop_x_axis_intersection_point: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = loop_x_axis_intersection_point
        self.b = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (((self.a + (self.b * self.base_points.list))/(np.power(1 + np.power(self.base_points.list,2),2))) + \
                1j * (((self.a + (self.b * self.base_points.list))/(np.power(1 + np.power(self.base_points.list,2),2))) * self.base_points.list)) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class BifoliumRegular(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * np.sin(self.base_points.list) * np.sin(2*(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class Biquartic(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size_mod: float,
        y_size_mod: float,
        starting_angle: float = 0,
        freq_mod: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.sin(3 * self.base_points.list) * np.cos(self.base_points.list)) + \
                1j * (self.b * np.power(np.sin(3 * self.base_points.list) * np.sin(self.base_points.list),2))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class BoothOvals(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        alpha: float,
        beta: float,
        starting_angle: float = 0,
        freq_mod: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = alpha
        self.b = beta

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (((self.a * np.power(self.b,2) * np.cos(self.base_points.list))/((np.power(self.b,2)*np.power(np.cos(self.base_points.list),2)) + (np.power(self.a,2)*np.power(np.sin(self.base_points.list),2)))) + \
                1j * ((np.power(self.a,2)*self.b*np.sin(self.base_points.list)) / ((np.power(self.b,2)*np.power(np.cos(self.base_points.list),2)) + (np.power(self.a,2)*np.power(np.sin(self.base_points.list),2))))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class BoothLemniscates(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        alpha: float,
        beta: float,
        starting_angle: float = 0,
        freq_mod: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = alpha
        self.b = beta

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (((self.b * np.sin(self.base_points.list)) / (1 + ((np.power(self.a,2))/(np.power(self.b,2)) * np.power(np.cos(self.base_points.list),2)))) + \
                1j *((self.a * np.sin(self.base_points.list) * np.cos(self.base_points.list)) / (1 + ((np.power(self.a,2))/(np.power(self.b,2)) * np.power(np.cos(self.base_points.list),2))))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Kiss(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size_mod: float,
        y_size_mod: float,
        starting_angle: float = 0,
        freq_mod: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.cos(self.base_points.list)) + \
                1j*(self.b * np.power(np.sin(self.base_points.list),3))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Clairaut(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        shape_mod: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.n = shape_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * np.power(np.sin(self.base_points.list),self.n)) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class ConchoidOfNicomedes(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        const_mod: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.b = const_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a / np.cos(self.base_points.list)) + self.b) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class Cornoid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size_mod: float,
        y_size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.cos(self.base_points.list) * np.cos(2*self.base_points.list)) + \
                1j * (self.b * np.sin(self.base_points.list) * (2 + np.cos(2*self.base_points.list)))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class MalteseCross(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size_mod: float,
        y_size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ( ((self.a * np.cos(self.base_points.list)) * (np.power(np.cos(self.base_points.list),2) - 2) ) + \
                1j*(self.b * np.sin(self.base_points.list) * np.power(np.cos(self.base_points.list),2))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class RationalCircularCubic(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        alpha: float,
        beta: float,
        delta: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = alpha
        self.b = beta
        self.d = delta

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ( ((self.d * np.power(self.base_points.list,2)) + (2*self.b*self.base_points.list) + (2*self.a) + self.d ) / (1 + np.power(self.base_points.list,2)) ) +\
                1j * (self.base_points.list * ( ((self.d * np.power(self.base_points.list,2)) + (2*self.b*self.base_points.list) + (2*self.a) + self.d ) / (1 + np.power(self.base_points.list,2)) )) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class SluzeCubic(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        alpha: float,
        beta: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = alpha
        self.b = beta

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a/np.cos(self.base_points.list)) + ((self.b*self.b/self.a) * np.cos(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class InvoluteCircle(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * (np.cos(self.base_points.list) + (self.base_points.list * np.sin(self.base_points.list)))) + \
                1j*(self.a * (np.sin(self.base_points.list) - (self.base_points.list * np.cos(self.base_points.list))))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class DumbBell(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        width: float,
        height: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = height
        self.b = width

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.cos(self.base_points.list)) + \
                1j*((np.power(self.a,2)*np.power(np.cos(self.base_points.list),2)*np.sin(self.base_points.list))/self.b)) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class StraightLine(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x2: float,
        y2: float,
        y1: float = 0,
        x1: float = 0,
        freq_mod: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=0,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()
        dx = (self.x2 - self.x1) / math.tau
        dy = (self.y2 - self.y1) / math.tau
        # dy = y2 - y1
        self.points = self.parent.points + \
            ((self.x1 + (dx * self.base_points.list)) + \
                1j * (self.y1 + (dy * self.base_points.list))) * \
            np.exp(1j * (self.freq_mod * self.base_points.list))
        # ((x1 + (dx * lop.point_array)) + 1j*(y1 + (dy * lop.point_array))) * \


class DurerShellCurve(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        alpha: float,
        beta: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = alpha
        self.b = beta

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((((self.a*np.cos(self.base_points.list))/(np.cos(self.base_points.list)-np.sin(self.base_points.list))) + (self.b*np.cos(self.base_points.list))) + \
                1j*(self.b*np.sin(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Ellipse(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        height: float,
        width: float,
        freq_mod: Union[float, np.ndarray] = 0,
        starting_angle: Union[float, np.ndarray] = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = width
        self.b = height

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.cos(self.base_points)) + \
                1j*(self.b*np.sin(self.base_points))) * \
            np.exp(1j * (self.starting_angle + self.freq_mod))
        # np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))
        # np.exp(1j * self.starting_angle)


class Rose(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        num_of_arms: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.n = num_of_arms

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * np.cos(self.n * (self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class Folioid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        alpha: float,
        eta: float,
        num_of_arms: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = alpha
        self.e = eta
        self.n = num_of_arms

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * (self.e * np.cos(self.n * (self.base_points.list)) + np.sqrt(1 - (np.power(self.e,2) * np.power(np.sin(self.n * (self.base_points.list)),2))))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class SimpleFolium(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a / np.power(1+np.power(self.base_points.list,2),2)) + \
                1j*((self.a / np.power(1+np.power(self.base_points.list,2),2))*self.base_points.list)) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Folium(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        radius: float,
        alpha: float,
        beta: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.r = radius
        self.a = alpha
        self.b = beta

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.r * np.cos(3*(self.base_points.list))) + (self.a * np.cos(self.base_points.list)) + (self.b * np.sin(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))
        # ((self.r * np.cos(3*(lop.point_array))) + (self.a * np.cos(lop.point_array)) + (self.b * (lop.point_array))) * \


class Cochleoid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        num_of_nodes: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.n = num_of_nodes + 2

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ( (self.a * np.sin((self.n*(self.base_points.list))/(self.n-1)) ) / (self.n*np.sin((self.base_points.list)/(self.n-1))) ) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class ConstantAngularAccelerationSpiral(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
        (self.a * np.power(self.base_points.list,2) / 2) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class GalileanSpiral(Curve):  #
    def __init__(
        self,
        parent: Type[Anchorable],
        alpha: float,
        beta: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = alpha
        self.b = beta

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
        (self.a + (self.b * np.power(self.base_points.list,2))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class HourGlass(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.sin(self.base_points.list)) + \
                1j*(self.a * np.sin(self.base_points.list) * np.cos(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Aerofoil(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        a: float,
        c: float,
        d: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = a
        self.c = c
        self.d = d

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        r = np.abs(self.c + (1j * self.d) - self.a)
        z0 = self.c + (1j * self.d) + (r * np.exp(1j *
                                                  (self.base_points.list)))
        self.points = self.parent.points + \
            ((1/2) * (z0 + (np.power(self.a,2)/z0))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class Piriform(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size: float,
        y_size: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size
        self.b = y_size

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * (1 + np.sin(self.base_points.list))) + \
                1j*(self.b * np.cos(self.base_points.list) * (1 + np.sin(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Teardrop(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        shape_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.n = shape_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.cos(self.base_points.list)) + \
                1j*(self.a * np.sin(self.base_points.list) * np.power((np.sin(self.base_points.list/2)),self.n))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class FigureEight(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (((self.a * np.sin(self.base_points.list))/(1 + np.power(np.cos(self.base_points.list),2))) + \
                1j * ((self.a*np.sin(self.base_points.list)*np.cos(self.base_points.list))/(1 + np.power(np.cos(self.base_points.list),2)))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Lissajous(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size: float,
        y_size: float,
        angle_mod: float,
        angle_shift: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size
        self.b = y_size
        self.n = angle_mod
        self.phi = angle_shift

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.sin(self.freq_mod * self.base_points.list)) + \
                1j*(self.b * np.sin((self.n*(self.freq_mod * self.base_points.list))+self.phi))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Egg(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        radius1: float,
        radius2: float,
        center_offset: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = radius1
        self.b = radius2
        self.d = center_offset

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (((np.sqrt((self.a*self.a) - (self.d*self.d*np.power(np.sin(self.base_points.list),2)))+(self.d*np.cos(self.base_points.list)))*np.cos(self.base_points.list)) + \
                1j*(self.b * np.sin(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class DoubleEgg(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * np.power(np.cos(self.base_points.list),2)) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class HabenichtTrefoil(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        num_of_pedals: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.n = num_of_pedals

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * (1 + np.cos(self.n*(self.base_points.list)) + np.power(np.sin(self.n*(self.base_points.list)),2))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class LaporteHeart(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size: float,
        y_size: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size
        self.b = y_size

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a*(np.power(np.sin(self.base_points.list),3))) + \
                1j*(self.b*(np.cos(self.base_points.list) - np.power(np.cos(self.base_points.list),4)))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class BossHeart(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size: float,
        y_size: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size
        self.b = y_size

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a*np.cos(self.base_points.list)) + \
                1j*(self.b*(np.sin(self.base_points.list) + np.sqrt(np.abs(np.cos(self.base_points.list)))))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Propeller(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * np.sqrt(np.sin(4 * (self.base_points.list)) / (1 - ((3/4) * np.power(np.sin(4 * (self.base_points.list)),2))) )) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class TFayButterfly(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        flair: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.b = flair

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a * (np.exp(np.cos(self.base_points.list)) - (2 * np.cos(4 * (self.base_points.list))) + (self.b * np.power(np.sin((self.base_points.list)/5),5)))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class Parabola(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        shape_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = shape_mod
        self.b = size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.b * ((self.base_points.list) + \
                1j * (self.a * np.power(self.base_points.list,2)))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Fish(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        fishiness: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.k = fishiness

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a*(np.cos(self.base_points.list) + 2*self.k*np.cos(self.base_points.list/2))) + \
                1j*(self.a*np.sin(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class DoubleFish(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size_mod: float,
        y_size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * ((5 * np.cos(self.base_points.list)) - ((np.sqrt(2)-1)*(np.cos(5* self.base_points.list))))) + \
                1j*(self.b * np.sin(4 * self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Polygasteroid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        eta: float,
        num_of_arms: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.e = eta
        self.n = num_of_arms

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a / (1 + (self.e * np.cos(self.n * self.freq_mod * self.base_points.list)))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class SinusoidLine(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        height_mod: float,
        wave_freq_mod: float = 0,
        rotational_freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=rotational_freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.c = size_mod
        self.a = height_mod
        self.b = 1 / wave_freq_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.c * ((self.base_points.list) + \
                1j * (self.a * np.sin(self.base_points.list/self.b)))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Sinusoid(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        wave_freq_mod: float = 1,
        rotational_freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=rotational_freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.b = wave_freq_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.sin(self.base_points.list * self.b)) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list))))


class BalanceSpring(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        size_mod: float,
        winding_constant: float,
        spring_stiffness: float,
        # freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            #  frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = size_mod
        self.k = winding_constant  # smaller num = more windings
        self.m = spring_stiffness  # larger num = less windings

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            (self.a / (1 + (self.k * np.exp(self.m * self.base_points.list)))) * \
            np.exp(1j * (self.starting_angle + (self.base_points.list)))


class Talbot(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size_mod: float,
        y_size_mod: float,
        c2: float = None,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size_mod
        self.b = y_size_mod
        if c2 is None:
            self.c2 = np.power(self.a, 2) - np.power(self.b, 2)
        else:
            self.c2 = c2
        # self.c2 = c2

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((  self.a * np.cos(self.base_points.list) * (1 + (self.c2 * np.power(np.sin(self.base_points.list),2) / np.power(self.a,2)) )) + \
                1j*(self.b * np.sin(self.base_points.list) * (1 - (self.c2 * np.power(np.cos(self.base_points.list),2) / np.power(self.b,2))) )) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class TNConstant(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size_mod: float,
        y_size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * (np.power(np.cos(self.base_points.list),2) + np.log(np.sin(self.base_points.list)))) + \
                1j*(self.b * np.sin(self.base_points.list) * np.cos(self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class CevaTrisectrix(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size_mod: float,
        y_size_mod: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size_mod
        self.b = y_size_mod

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a *(np.cos(3*self.base_points.list) + (2*np.cos(self.base_points.list)))) + \
                1j*(self.b * np.sin(3 * self.base_points.list))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


class Basin2D(Curve):
    def __init__(
        self,
        parent: Type[Anchorable],
        x_size_mod: float,
        y_size_mod: float,
        num_of_pedals: float,
        pedal_rotation: float,
        freq_mod: float = 0,
        starting_angle: float = 0,
        deg: bool = False,
        steps_per_cycle: int = 1000,
        cycle_start: float = 0,
        cycle_end: float = 10,
        num_of_cycles: int = 1,
    ):
        super().__init__(
            parent=parent,
            starting_angle=starting_angle,
            frequency=freq_mod,
            deg=deg,
            steps_per_cycle=steps_per_cycle,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
            num_of_cycles=num_of_cycles,
        )
        self.a = x_size_mod
        self.b = y_size_mod
        self.n = num_of_pedals
        self.phi = pedal_rotation

    def create_point_lists(self) -> None:
        #super().create_point_lists()
        self.parent.create_point_lists()

        self.points = self.parent.points + \
            ((self.a * np.cos(self.base_points.list + self.phi) * np.cos(self.n*self.base_points.list)) +\
                1j*(self.b * np.power(np.cos(self.n * self.base_points.list),2))) * \
            np.exp(1j * (self.starting_angle + (self.freq_mod * self.base_points.list)))


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

        self.helper_class: Union[TwoBarFixedLinkage, None] = None

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

    def update_drawing_objects(self, frame) -> None:
        if self.helper_class is None:
            raise FreeBarHelperClassError(
                "FreeBar used without helper class to update drawing objects")
        else:
            self.helper_class.update_drawing_objects(frame)

    def create_point_lists(self) -> None:
        if self.helper_class is None:
            raise FreeBarHelperClassError(
                "FreeBar used without helper class to initialize points array")
        else:
            self.helper_class.create_point_lists()


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
        self.parent2mate_dotted_line.set_data(
            [self.parent.points[frame].real, self.mate.points[frame].real],
            [self.parent.points[frame].imag, self.mate.points[frame].imag])

        self.parent2link_solid_line.set_data([
            self.parent.points[frame].real, self.link_point_array[frame].real
        ], [self.parent.points[frame].imag, self.link_point_array[frame].imag])

        self.link2arm_solid_line.set_data(
            [self.link_point_array[frame].real, self.points[frame].real],
            [self.link_point_array[frame].imag, self.points[frame].imag])

    def create_point_lists(self) -> None:
        if self.points is None:
            self.parent.create_point_lists()
            self.mate.create_point_lists()

            self.link_point_array = self.parent.points + \
                (self.link_point_length * np.exp(1j * np.angle(self.mate.points - self.parent.points)))

            self.points = self.link_point_array + \
                (self.arm_length * np.exp(1j * (np.angle(self.mate.points - self.parent.points) + self.arm_angle)))

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
        if self.points is None:
            self.parent.create_point_lists(foundation_list_of_points)
            self.mate.create_point_lists(foundation_list_of_points)

            self.link_point_length = self.link_point_percentage * (np.sqrt(
                np.power(self.mate.points[:].real -
                         self.parent.points[:].real, 2) +
                np.power(self.mate.points[:].imag -
                         self.parent.points[:].imag, 2)))
            super().create_point_lists(foundation_list_of_points)

            # self.link_point_array = self.parent.point_array + \
            #     ((self.link_point_percentage * (self.mate.point_array - self.parent.point_array)) * \
            #     np.exp(1j * np.angle(self.mate.point_array - self.parent.point_array)))

            # self.point_array = self.link_point_array + \
            #     (self.arm_length * np.exp(1j * (np.angle(self.mate.point_array - self.parent.point_array) + self.arm_angle)))


class TwoBarFixedLinkage(Anchorable):
    def __init__(self,
                 primary_linkage: FreeBar,
                 secondary_linkage: FreeBar,
                 flip_intersection: bool = False):

        super().__init__()
        self.primary_linkage: FreeBar = primary_linkage
        self.secondary_linkage: FreeBar = secondary_linkage
        self.linkage_point_array: Union[np.ndarray, None] = None
        self.flip_intersection: bool = flip_intersection

        self.primary_linkage.helper_class = self
        self.secondary_linkage.helper_class = self

    def create_point_lists(self) -> None:
        if self.points is None:
            self.primary_linkage.parent.create_point_lists()
            self.secondary_linkage.parent.create_point_lists()

            x0 = self.primary_linkage.parent.points.real
            y0 = self.primary_linkage.parent.points.imag
            r0 = self.primary_linkage.link_point_length

            x1 = self.secondary_linkage.parent.points.real
            y1 = self.secondary_linkage.parent.points.imag
            r1 = self.secondary_linkage.link_point_length

            # if x0[0] > x1[0]:
                # x0, y0, r0, x1, y1, r1 = x1, y1, r1, x0, y0, r0


            # print(f'x1-x0**2: {np.power(x1-x0,2)}')
            # print(f'y1-y0**2: {np.power(y1-y0,2)}')
            d = np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))

            # print(f'r0: {r0}')
            # print(f'r1: {r1}')
            # print(f'd: {np.power(d,2)}')
            # print(f'r1^2: {np.power(r1,2)}')
            # print(f'd^2: {np.power(d,2)}')
            # print(f'comb: {(np.power(r0, 2) - np.power(r1, 2) + np.power(d, 2))}')
            # print(f'a: {(np.power(r0, 2) - np.power(r1, 2) + np.power(d, 2)) / (2 * d)}')
            a = (np.power(r0, 2) - np.power(r1, 2) + np.power(d, 2)) / (2 * d)
            # h = np.sqrt(np.power(r0, 2) - np.power(a, 2))
            # print(f'r0^2: {np.power(r0,2)}')
            # print(f'a:  {np.power(a,2)}')
            h = np.sqrt((r0**2) - (a**2))

            # print(f'h: {h}')

            x2 = x0 + (a * (x1 - x0)) / d
            y2 = y0 + (a * (y1 - y0)) / d

            if self.flip_intersection:
                x4 = x2 + (h * (y1 - y0)) / d
                y4 = y2 - (h * (x1 - x0)) / d
            else:
                x4 = x2 - (h * (y1 - y0)) / d
                y4 = y2 + (h * (x1 - x0)) / d

            #! Gross, manually setting primary and secondary linkage point array feels shitty

            self.linkage_point_array = x4 + (1j * y4)
            # TEMP
            # print(f'linkage points: {self.linkage_point_array}')

            self.secondary_linkage.points = self.linkage_point_array + \
                (self.secondary_linkage.arm_length * \
                np.exp(1j * (np.angle(self.linkage_point_array - self.secondary_linkage.parent.points) + self.secondary_linkage.arm_angle)))

            self.points = self.linkage_point_array + \
                (self.primary_linkage.arm_length * \
                np.exp(1j * (np.angle(self.linkage_point_array - self.primary_linkage.parent.points) + self.primary_linkage.arm_angle)))

            self.primary_linkage.points = self.points

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
            self.secondary_linkage.parent2link_solid_line,
            self.secondary_linkage.link2arm_solid_line
        ]

    def update_drawing_objects(self, frame) -> None:
        self.primary_linkage.parent2link_solid_line.set_data([
            self.primary_linkage.parent.points[frame].real,
            self.linkage_point_array[frame].real
        ], [
            self.primary_linkage.parent.points[frame].imag,
            self.linkage_point_array[frame].imag
        ])
        self.primary_linkage.link2arm_solid_line.set_data(
            [self.linkage_point_array[frame].real, self.points[frame].real],
            [self.linkage_point_array[frame].imag, self.points[frame].imag])
        self.secondary_linkage.parent2link_solid_line.set_data([
            self.secondary_linkage.parent.points[frame].real,
            self.linkage_point_array[frame].real
        ], [
            self.secondary_linkage.parent.points[frame].imag,
            self.linkage_point_array[frame].imag
        ])
        self.secondary_linkage.link2arm_solid_line.set_data([
            self.linkage_point_array[frame].real,
            self.secondary_linkage.points[frame].real
        ], [
            self.linkage_point_array[frame].imag,
            self.secondary_linkage.points[frame].imag
        ])

    def get_linkage_objects(self):
        return self.primary_linkage, self.secondary_linkage


a1 = Anchor()
main_circle_radius = ListOfPoints(1000, 800, 800, treat_as_length_or_radius=True)
main_circle_radius = SinglePointList(900)
base = 1
mc1 = Circle(
    a1,
    main_circle_radius,
    starting_angle=0,
    cycle_end=base,
    # steps_per_cycle=10
)
mc2 = Circle(
    a1,
    main_circle_radius,
    starting_angle=math.tau/12,
    cycle_end=base,
    # steps_per_cycle=10
)
sc2_ce = -base*90
sc1_ce = base*139

sc1 = Circle(
    mc1,
    SinglePointList(140),
    cycle_end=sc1_ce,
    starting_angle=1
    # steps_per_cycle=10
)

sc2 = Circle(
    mc2,
    SinglePointList(180),
    cycle_end=sc2_ce,
    # steps_per_cycle=10
)

tbl1 = TwoBarFixedLinkage(
    FreeBar(sc1,450,120),
    FreeBar(sc2,500,120),
    flip_intersection=False
)

tbl1_fb1, tbl2_fb2 = tbl1.get_linkage_objects()

tbl2 = TwoBarFixedLinkage(
    FreeBar(tbl1_fb1, 175),
    FreeBar(tbl2_fb2, 175),
    flip_intersection=True
)
tbl2.export_gcode(point_array_starting_offset=0, decimal_precision=2, file_name="CCCFFFF-3", canvas_xlimit=205, canvas_ylimit=270)

# tbl2.animate(speed=20)
# tbl1.animate(speed=60)
# sc1.animate(speed=60)
# mc1.animate(speed=70)

# a1 = Anchor(-350 + 70j)
# # a2 = Anchor(150 + -200j)

# ac1 = Circle(1400, 0.1, parent=a1, starting_angle=0)
# ac2 = Circle(1400, 0.1, parent=a1, starting_angle=math.tau / 16)

# c1 = Circle(200, -0.33333, parent=ac1)
# c2 = Circle(200, 1.4, parent=ac2)

# # b1 = Bar(c1, 300, c2, 150, arm_length=100, arm_angle=math.tau/4)
# # b1 = Bar(c1, 500, c2, arm_length=450)
# # b1 = Point2PointSliderBar(c1, c2, 450, 100, math.tau/8)
# # b1 = Point2PointElasticBar(c1, c2, 50, 200, math.tau / 8)
# l1 = TwoBarLinkage(FreeBar(c1, 550, 300),
#                    FreeBar(c2, 600, 300),
#                    flip_intersection=True)
# b1, b2 = l1.get_linkage_objects()

# l2 = TwoBarLinkage(FreeBar(b1, 300, 200),
#                    FreeBar(b2, 300, 100),
#                    flip_intersection=True)

# c1 = Circle(500, starting_angle=1, cycle_end=60, num_of_cycles=1, steps_per_cycle=100)
# c2 = Circle(300, parent=c1, starting_angle=3, cycle_end=121, num_of_cycles=1)
# c3 = Circle(200, parent=c2, starting_angle=1.3, cycle_end=180, num_of_cycles=1)

# freq2=66 * base
# freq3=100 * base

# lop3 = ListOfPoints(3000, 10, 10)

# a1 = Anchor(0 + 0j)
# base=60
# c3_rot = base*(11+0.00)
# c4_rot = base*(2+ 0.08)

# c1_radius = SinglePointList(600 * math.tau)
# c1 = Circle(a1, c1_radius, cycle_end=base)
# c2 = Circle(a1, c1_radius, math.tau*(2/3), cycle_end=base)

# c3_radius = SinglePointList(150 * math.tau)
# c3 = Circle(c1, c3_radius, cycle_end=c3_rot)

# c4_rad = SinglePointList(100 * math.tau)
# c4 = Circle(c3, c4_rad, cycle_end=c4_rot)

# b1 = Point2PointSliderBar(c4,c2,1020,530, math.tau*(7/38))

# b1.animate(speed=20)
# ecycle=base*190
# sizecycles=10
# rot  = ListOfPoints(500, 0, 1, num_of_cycles=1)
# lop1 = ListOfPoints(1, 150, 150, num_of_cycles=1)
# lop2 = ListOfPoints(10, 5,   140, num_of_cycles=sizecycles)
# lop3 = ListOfPoints(50, 25,   12, num_of_cycles=sizecycles)
# c1 = Circle(a1, lop1,            starting_angle=-0.4, cycle_end=1,         num_of_cycles=1, steps_per_cycle=500)
# e1 = Ellipse(c1,lop2,lop3, starting_angle=(math.tau/4)-0.4 ,cycle_end=ecycle, freq_mod=rot, steps_per_cycle=100)
# lop3 = ListOfPoints(1000, 10,   60, num_of_cycles=16)
# e1 = Ellipse(a1,lop2,lop3, starting_angle=math.tau/4 ,cycle_end=ecycle, freq_mod=0.01, steps_per_cycle=1000)
# e1 = Ellipse(a1,lop2,lop3, starting_angle=math.tau/2 ,cycle_end=ecycle, steps_per_cycle=1000)
# s1 = Sinusoid(c1, size_mod=lop2, wave_freq_mod=9, steps_per_cycle=1000, rotational_freq_mod=base/10)
# c2 = Circle(lop2, parent=c1, starting_angle=2, cycle_end=freq2, num_of_cycles=1, steps_per_cycle=3000)
# e2 = Ellipse(c1,)A
# c3 = Circle(lop3, parent=c2, starting_angle=3, cycle_end=freq3, num_of_cycles=1, steps_per_cycle=3000)
# s1 = ArchimedeanSpiral(c2, 0, 0, 20, cycle_end=1

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
