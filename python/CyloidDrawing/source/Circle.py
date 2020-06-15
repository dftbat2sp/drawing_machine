import math
import cmath
from typing import Type, Union, Tuple, Iterable, List
import numpy as np
from dataclasses import dataclass
import itertools

import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
import matplotlib.gridspec as gridspec


class IntersectionError(Exception):
    pass


"""
def get_circles_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

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
        h = math.sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        return x3 + y3 * 1j
"""


@dataclass
class RotationResolution:
    step_size: float = 0.001
    rotations: float = 50

    def __post_init__(self):
        self.num_of_points: int = int(self.rotations / self.step_size)
        self.rotation_to_radians: float = self.rotations * math.tau
        self.point_list: np.ndarray = np.linspace(0, self.rotation_to_radians, self.num_of_points)


class Anchorable:

    def __init__(self):
        self.point_array: Type[np.ndarray, None] = None
        self.list_calculated = False

    def create_point_lists(self, base_points_list: RotationResolution) -> None:
        raise NotImplementedError

    def update_drawing_objects(self, frame) -> None:
        raise NotImplementedError

    def get_main_drawing_objects(self) -> List:
        raise NotImplementedError

    def get_secondary_drawing_objects(self) -> List:
        raise NotImplementedError

    def get_min_max_values(self, buffer: float = 0, point_array_only: bool = False) -> Tuple:
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
        raise NotImplementedError

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
        self.main_marker = plt.Line2D([self.point.real], [self.point.imag], marker="^")
        self.secondary_marker = plt.Line2D([self.point.real], [self.point.imag], marker="^")

    def create_point_lists(self, base_points_list: RotationResolution):
        self.point_array = np.full_like(base_points_list.point_list, self.point, dtype=complex)
        # self.point_array = self.point

    def update_drawing_objects(self, frame) -> None:
        pass  # do nothing, point stays still

    def get_main_drawing_objects(self) -> List:
        return [self.main_marker, ]

    def get_secondary_drawing_objects(self) -> List:
        return [self.secondary_marker, ]

    def get_min_max_values(self, buffer: float = 0, point_array_only: bool = False) -> Tuple:
        return self.point - buffer, self.point + buffer, self.point - buffer, self.point + buffer

    def get_min_max_values_normalized_to_origin(self, buffer: float = 0) -> Tuple:
        return 0 - buffer, 0 + buffer, 0 - buffer, 0 + buffer

    def __add__(self, other):
        return Anchor(self.point + other.point)


class Circle(Anchorable, BarMateSlide):
    exp_const = math.tau * 1j

    def __init__(self,
                 radius: float,
                 frequency: float,
                 starting_angle: float = 0,
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
        self.length_starting_angle: complex = self.radius * cmath.exp(self.starting_angle * 1j)

        self.parent: Anchorable = parent

        """
        Drawing Objects
        """
        self.main_circle_edge_artist = plt.Circle((0, 0), self.radius, fill=False)
        self.main_centre2point_line_artist = plt.Line2D([], [], marker='.', markevery=(1, 1))

        self.secondary_circle_edge_artist = plt.Circle((0, 0), self.radius, fill=False)
        self.secondary_centre2point_line_artist = plt.Line2D([], [], marker='.', markevery=(1, 1))

        super().__init__()

    def create_point_lists(self, base_points_list: RotationResolution) -> None:
        if self.point_array is None:
            self.parent.create_point_lists(base_points_list)

            self.point_array = self.parent.point_array + self.length_starting_angle * np.exp(
                self.exp_const * self.rotation_frequency * base_points_list.point_list)

    def update_drawing_objects(self, frame) -> None:
        # MAIN
        self.main_circle_edge_artist.set_center(
            (self.parent.point_array[frame].real, self.parent.point_array[frame].imag))
        self.main_centre2point_line_artist.set_data(
            [self.parent.point_array[frame].real, self.point_array[frame].real],  # x
            [self.parent.point_array[frame].imag, self.point_array[frame].imag])  # y

        # SECONDARY
        self.secondary_centre2point_line_artist.set_data(
            [0, self.point_array[frame].real - self.parent.point_array[frame].real],  # x
            [0, self.point_array[frame].imag - self.parent.point_array[frame].imag])  # y

    def get_main_drawing_objects(self) -> List:
        return [self.main_circle_edge_artist, self.main_centre2point_line_artist]

    def get_secondary_drawing_objects(self) -> List:
        return [self.secondary_circle_edge_artist, self.secondary_centre2point_line_artist]

    def get_min_max_values(self, buffer: float = 0, point_array_only: bool = False) -> Tuple:
        x_min = min(self.point_array[:].real) - buffer
        x_max = max(self.point_array[:].real) + buffer

        y_min = min(self.point_array[:].imag) - buffer
        y_max = max(self.point_array[:].imag) + buffer

        return x_min, x_max, y_min, y_max

    def get_min_max_values_normalized_to_origin(self, buffer=0) -> Tuple:
        x_max = self.radius + buffer
        x_min = -1 * x_max

        y_max = x_max
        y_min = x_min

        return x_min, x_max, y_min, y_max

    def __str__(self):
        return f'radius: {self.radius}, freq: {self.rotation_frequency}, angle: {self.starting_angle}'


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
        self.mate: Type[Anchorable, BarMateSlide, BarMateFix] = mate_object
        self.point_length = point_length_from_parent
        self.mate_length = mate_length_from_parent
        self.arm_length: float = arm_length
        self.arm_angle: float = arm_angle
        self.mate_point_array = None
        if deg:
            self.arm_angle = np.radians(self.arm_angle)

        self.pre_arm_point_array: Type[np.ndarray, None] = None

        """
        Drawing Objects
        """
        self.main_mate_line = plt.Line2D([], [], marker='D', markevery=(1, 1), linestyle='--')
        self.main_pre_arm_point_line = plt.Line2D([], [], marker='x', markevery=(1, 1))
        self.main_arm_line = plt.Line2D([], [], marker='.', markevery=(1, 1))
        self.mate_intersection_circle = plt.Circle((0, 0), self.mate_length, fill=False)

        self.secondary_mate_line = plt.Line2D([], [], marker='D', markevery=(1, 1), linestyle='--')
        self.secondary_pre_arm_point_line = plt.Line2D([], [], marker='x', markevery=(1, 1))
        self.secondary_arm_line = plt.Line2D([], [], marker='x', markevery=(1, 1))

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
            [self.parent.point_array[frame].real, self.mate_point_array[frame].real],
            [self.parent.point_array[frame].imag, self.mate_point_array[frame].imag]
        )
        self.main_pre_arm_point_line.set_data(
            [self.parent.point_array[frame].real, self.pre_arm_point_array[frame].real],
            [self.parent.point_array[frame].imag, self.pre_arm_point_array[frame].imag]
        )
        self.main_arm_line.set_data(
            [self.pre_arm_point_array[frame].real, self.point_array[frame].real],
            [self.pre_arm_point_array[frame].imag, self.point_array[frame].imag],
        )

        # self.mate_intersection_circle.set_center(
        #     (self.parent.point_array[frame].real, self.parent.point_array[frame].imag))

        # SECONDARY
        self.secondary_mate_line.set_data(
            [0, self.mate_point_array[frame].real - self.parent.point_array[frame].real],
            [0, self.mate_point_array[frame].imag - self.parent.point_array[frame].imag]
        )
        self.secondary_pre_arm_point_line.set_data(
            [0, self.pre_arm_point_array[frame].real - self.parent.point_array[frame].real],
            [0, self.pre_arm_point_array[frame].imag - self.parent.point_array[frame].imag]
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

    def get_min_max_values(self, buffer: float = 0, point_array_only: bool = False) -> Tuple:

        real_list = [self.point_array[:].real]
        imag_list = [self.point_array[:].imag]
        if not point_array_only:
            real_list.append(self.mate_point_array[:].real)
            real_list.append(self.parent.point_array[:].real)
            real_list.append(self.pre_arm_point_array[:].real)

            imag_list.append(self.mate_point_array[:].imag)
            imag_list.append(self.parent.point_array[:].imag)
            imag_list.append(self.pre_arm_point_array[:].imag)

        x_min = min(itertools.chain.from_iterable(real_list)) - buffer
        x_max = max(itertools.chain.from_iterable(real_list)) + buffer

        y_min = min(itertools.chain.from_iterable(imag_list)) - buffer
        y_max = max(itertools.chain.from_iterable(imag_list)) + buffer

        return x_min, x_max, y_min, y_max

    # TODO
    # should parent be normalized or another point? MATE?
    def get_min_max_values_normalized_to_origin(self, buffer: float = 0) -> Tuple:
        x_min = min(itertools.chain(self.point_array[:].real - self.parent.point_array[:].real,
                                    self.mate_point_array[:].real - self.parent.point_array[:].real,
                                    self.pre_arm_point_array[:].real - self.parent.point_array[:].real,
                                    [0])) - buffer
        x_max = max(itertools.chain(self.point_array[:].real - self.parent.point_array[:].real,
                                    self.mate_point_array[:].real - self.parent.point_array[:].real,
                                    self.pre_arm_point_array[:].real - self.parent.point_array[:].real,
                                    [0])) + buffer

        y_min = min(itertools.chain(self.point_array[:].imag - self.parent.point_array[:].imag,
                                    self.mate_point_array[:].imag - self.parent.point_array[:].imag,
                                    self.pre_arm_point_array[:].imag - self.parent.point_array[:].imag,
                                    [0])) - buffer
        y_max = max(itertools.chain(self.point_array[:].imag - self.parent.point_array[:].imag,
                                    self.mate_point_array[:].imag - self.parent.point_array[:].imag,
                                    self.pre_arm_point_array[:].imag - self.parent.point_array[:].imag,
                                    [0])) + buffer

        return x_min, x_max, y_min, y_max

    # TODO fix circle choice (x3,y3) vs (x4,y) so that cross beam all point and and not OUT IN

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

            """
            d: np.ndarray = np.sqrt(
                np.power((self.parent.point_array.real - self.mate.parent.point_array.real), 2) +
                np.power((self.parent.point_array.imag - self.mate.parent.point_array.imag), 2))

            if (d > (self.mate_length + self.mate.mate_length)).any():
                raise IntersectionError(f'Non-intersecting, non-concentric circles not contained within each other.')
            elif (d < abs(self.mate_length - self.mate.mate_length)).any():
                raise IntersectionError(f'Non-intersecting circles. One contained in the other.')
            elif (d == 0).any():
                raise IntersectionError(f'Concentric circles.')
            else:
                """
            # doing the below calc in fewer varialbes to save on memory
            # a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
            # h = sqrt(r0 ** 2 - a ** 2)
            # x2 = x0 + a * (x1 - x0) / d
            # y2 = y0 + a * (y1 - y0) / d
            # x3 = x2 + h * (y1 - y0) / d
            # y3 = y2 - h * (x1 - x0) / d
            """
            # these variables kept out of larger equation for ease of reading and deduplicate processing the equation multiple times
            a = (np.power(self.mate.mate_length, 2) - np.power(self.mate_length, 2) + np.power(d, 2)) / (2 * d)
            h = np.sqrt(np.power(self.mate.mate_length, 2) - np.power(a, 2))

            # @formatter:off
            self.mate_point_array = (
                (self.parent.point_array.real + (a * (
                        self.parent.point_array.real - self.mate.parent.point_array.real) / d)) - (
                    h * (self.parent.point_array.imag - self.mate.parent.point_array.imag) / d)
                    ) + (
                1j * (
                (self.parent.point_array.imag + (a * (
                        self.parent.point_array.imag - self.mate.parent.point_array.imag) / d)) + (
                    h * (self.parent.point_array.real - self.mate.parent.point_array.real) / d)))
            # (self.parent.point_array.real + (a * (
            #         self.mate.parent.point_array.real - self.parent.point_array.real) / d)) + (
            #         h * (self.mate.parent.point_array.imag - self.parent.point_array.imag) / d)
            # ) + (1j * (
            #         (self.parent.point_array.imag + (a * (
            #                 self.mate.parent.point_array.imag - self.parent.point_array.imag) / d)) - (
            #                 h * (self.mate.parent.point_array.real - self.parent.point_array.real) / d)))
            # @formatter:on    
            """

            self.mate.mate_point_array = self.mate_point_array

        elif self.mate.mate_point_array is None:
            self.mate.mate_point_array = self.mate_point_array


def animate_all(drawer: Anchorable, resolution_obj: RotationResolution, *components: Anchorable):
    """Create point list for drawer and all subcomponents"""
    drawer.create_point_lists(resolution_obj)

    for comp in components:
        comp.create_point_lists(resolution_obj)

    """Get mpl.artist objects for main drawing"""
    main_plot_drawer_artist_list = drawer.get_main_drawing_objects()
    main_plot_component_artist_list = []

    for comp in components:
        main_plot_component_artist_list.extend(comp.get_main_drawing_objects())

    """ Create basic figure and axes """
    fig = plt.figure(figsize=(21, 13))

    gs_parent_num_of_rows = 1
    gs_parent_num_of_cols = 2
    gs_parent = gridspec.GridSpec(gs_parent_num_of_rows, gs_parent_num_of_cols, figure=fig)

    num_of_component_plot_rows = 2
    num_of_component_plot_cols = int(math.ceil(len(components) / num_of_component_plot_rows))

    print(f'plot component_col: {num_of_component_plot_cols}')

    gs_component_plots = gridspec.GridSpecFromSubplotSpec(num_of_component_plot_rows, num_of_component_plot_cols,
                                                          subplot_spec=gs_parent[0])

    gs_main_plot_num_of_rows = 1
    gs_main_plot_num_of_cols = 2
    gs_main_plots = gridspec.GridSpecFromSubplotSpec(gs_main_plot_num_of_rows, gs_main_plot_num_of_cols,
                                                     subplot_spec=gs_parent[1])

    main_animated_draw_axis = fig.add_subplot(gs_main_plots[:, 0])
    main_final_shape_axis = fig.add_subplot(gs_main_plots[:, 1])

    """ min/max for main drawer axes """
    animated_space_buffer = 1
    animated_drawer_x_min, animated_drawer_x_max, animated_drawer_y_min, animated_drawer_y_max = drawer.get_min_max_values(
        buffer=animated_space_buffer,
        point_array_only=False)
    final_space_buffer = 0.2
    final_drawer_x_min, final_drawer_x_max, final_drawer_y_min, final_drawer_y_max = drawer.get_min_max_values(
        buffer=final_space_buffer,
        point_array_only=True)

    # set xy limits for final drawing axes
    main_animated_draw_axis.set_xlim((animated_drawer_x_min, animated_drawer_x_max))
    main_animated_draw_axis.set_ylim((animated_drawer_y_min, animated_drawer_y_max))

    main_final_shape_axis.set_xlim((final_drawer_x_min, final_drawer_x_max))
    main_final_shape_axis.set_ylim((final_drawer_y_min, final_drawer_y_max))

    # set axis aspect ratio for final drawing axes
    main_animated_draw_axis.set_aspect('equal')
    main_final_shape_axis.set_aspect('equal')

    main_animated_drawer_final_line, = main_animated_draw_axis.plot([], [])
    # finished_final_line, = main_final_shape_axis.plot(drawer.point_array.real, drawer.point_array.imag)
    main_final_shape_axis.plot(drawer.point_array.real, drawer.point_array.imag)

    """ What's going on here? """
    comp_axis_artist_list = []
    individual_axis_space_buffer = 0.5

    for num, comp in enumerate(components):
        component_col = int(math.floor(num / num_of_component_plot_rows))
        component_row = num % num_of_component_plot_rows

        component_x_min, component_x_max, component_y_min, component_y_max = comp.get_min_max_values_normalized_to_origin(
            buffer=individual_axis_space_buffer)

        temp_ax = fig.add_subplot(gs_component_plots[component_row, component_col])
        temp_ax.set_xlim((component_x_min, component_x_max))
        temp_ax.set_ylim((component_y_min, component_y_max))
        temp_ax.set_aspect('equal')

        for artist in comp.get_secondary_drawing_objects():
            # add artist to axix
            temp_ax.add_artist(artist)
            # add artist to component artist list for figure (needed for animation)
            comp_axis_artist_list.append(artist)

    fig.tight_layout()

    def on_q(event):
        if event.key == 'q':
            exit()

    def init():
        for artist in itertools.chain(main_plot_drawer_artist_list, main_plot_component_artist_list):
            main_animated_draw_axis.add_artist(artist)

        return itertools.chain([main_animated_drawer_final_line],
                               main_plot_drawer_artist_list,
                               main_plot_component_artist_list,
                               comp_axis_artist_list)

    def get_frames():
        for i in range(drawer.point_array.size):
            point = i * 1
            if point < drawer.point_array.size:
                yield point

    def animate(frame):

        drawer.update_drawing_objects(frame)

        for anchorable_obj in components:
            anchorable_obj.update_drawing_objects(frame)
        # frame = frame - 1
        main_animated_drawer_final_line.set_data(drawer.point_array[:frame + 1].real,
                                                 drawer.point_array[:frame + 1].imag)

        return itertools.chain([main_animated_drawer_final_line], main_plot_drawer_artist_list,
                               main_plot_component_artist_list,
                               comp_axis_artist_list)

    cid = fig.canvas.mpl_connect('key_press_event', on_q)

    mpl_animation.FuncAnimation(fig, animate,
                                init_func=init,
                                interval=1,
                                blit=True,
                                save_count=1,
                                frames=get_frames,
                                repeat=False)

    plt.show()


""" Draw """
"""
anchor1 = Anchor(1 + 1j)
circle1 = Circle(1, 0.233, parent_object=anchor1)
circle2 = Circle(0.35, 0.377, parent_object=circle1)
circle3 = Circle(0.1, 2, parent_object=circle2)
circle4 = Circle(0.2, 0.01, parent_object=circle3)
circle5 = Circle(0.2, 0.01, starting_angle=1, parent_object=circle4)
"""
"""
anchor1 = Anchor(-2 - 1j)
anchor2 = Anchor(1 + 2j)

circle1 = Circle(0.5, 1.2, parent=anchor1)
circle2 = Circle(0.6, 1.23, parent=anchor2)

bar1 = Bar(circle1, 6, circle2)

base_points = RotationResolution(rotations=5)
# anchor1.create_point_lists(pointlist)

# animate_all(circle1, base_points, anchor1)
# animate_all(circle2, base_points, anchor1, circle1)
# animate_all(circle3, base_points, anchor1, circle1, circle2)
# animate_all(circle4, base_points, anchor1, circle1, circle2, circle3)
# animate_all(circle5, base_points, circle1, circle2, circle3, circle4)
animate_all(bar1, base_points, circle1, circle2)
"""

base_points = RotationResolution(rotations=10, step_size=0.0005)
# 150 / 100
# middle_rotation = 120 / 100
middle_rotation = 3/2
outer_rotation = 84/8

circle_middle_circle = Circle(4, middle_rotation, starting_angle=-2*np.pi*(1/8))
# circle_middle_circle = Circle(3, middle_rotation, starting_angle=-2*np.pi*(1/8))
circle_middle_anchor = Circle(4, middle_rotation, starting_angle=2*np.pi*(4/8))

# circle_outside = Circle(1, middle_rotation, parent=circle_middle_circle)
circle_outside = Circle(2.5, outer_rotation, parent=circle_middle_circle)

# bar_draw = Bar(circle_outside, 2.2, circle_middle_anchor, arm_angle=-np.pi/2, arm_length=2)
# bar_draw = Bar(circle_middle_anchor, 3, circle_outside, arm_angle=np.pi/2, arm_length=0.2)

# animate_all(bar_draw, base_points, circle_middle_circle, circle_outside)
animate_all(circle_outside, base_points, circle_middle_anchor, circle_middle_circle, circle_outside)



"""
base_points = RotationResolution(rotations=5, step_size=0.0001)
# anchor2 = Anchor(4+0j)
# circle1 = Circle(0.6, 8)
# circle2 = Circle(0.8, 2, parent_object=anchor2)
# circle3 = Circle(0.3, 20, parent_object=circle2)
# anchor4 = Anchor(-3 - 1j)

# center_rotation = -10/24

anchor1 = Anchor(-4)
anchor2 = Anchor(4)
drive_circle1 = Circle(3, 0.5, parent=anchor1)
drive_circle2 = Circle(0.5, 113, parent=anchor2)
# rotation_circle1 = Circle(3, 0.1, starting_angle=0)  # right
# rotation_circle2 = Circle(3, 0.1, starting_angle=np.pi)  # left
# drive_circle1 = Circle(2, 1 / 13, parent=rotation_circle1, starting_angle=np.pi)
# drive_circle2 = Circle(2, 1 / 17, parent=rotation_circle2)
bar1_1 = Bar(drive_circle1, 11, None, mate_length_from_parent=6)  # right
bar1_2 = Bar(drive_circle2, 11, None, mate_length_from_parent=6)  # left
bar1_2.mate = bar1_1
bar1_1.mate = bar1_2

bar2_1 = Bar(bar1_1, 6, None, mate_length_from_parent=6)
bar2_2 = Bar(bar1_2, 6, bar2_1, mate_length_from_parent=6)
bar2_1.mate = bar2_2

animate_all(bar2_2, base_points)
"""
"""
x3 = x2 + h * (self.mate.parent.point_array.imag - self.parent.point_array.imag) / d
y3 = y2 - h * (self.mate.parent.point_array.real - self.parent.point_array.real) / d



x3 = x2 + h * (
        y1 - y0) / d
y3 = y2 - h * (x1 - x0) / d

self.mate_point_array = \
    ((self.parent.point_array.real + (
                (self.mate_length ** 2 - self.mate.mate_length ** 2 + d ** 2) / (2 * d)) * (
                  self.mate.parent.point_array.real - self.parent.point_array.real) / d) + (
         sqrt(self.mate_length ** 2 - (
                     (self.mate_length ** 2 - self.mate.mate_length ** 2 + d ** 2) / (2 * d)) ** 2)) * (
             self.mate.parent.point_array.imag - self.parent.point_array.imag) / d) + \
    ((self.parent.point_array.imag + (
                (self.mate_length ** 2 - self.mate.mate_length ** 2 + d ** 2) / (2 * d)) * (
                  self.mate.parent.point_array.imag - self.mate.parent.point_array.imag) / d) - (
         sqrt(self.mate_length ** 2 - (
                     (self.mate_length ** 2 - self.mate.mate_length ** 2 + d ** 2) / (2 * d)) ** 2)) * (
             self.mate.parent.point_array.real - self.parent.point_array.real) / d) * 1j

self.mate.mate_point_array = self.mate_point_array
"""
"""
class Draw:

    def __init__(self, draw_obj, resolution, num_of_rotations, *supporting_objs):

        self.draw_obj = draw_obj
        self.supporting_obj = supporting_objs

        self.mate_lengthesolution = resolution
        self.num_of_rotations = num_of_rotations

"""
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

# n1 = [b1.get_point(i) for i in range(20000000)]

print('Done')
import time

time.sleep(20)
# Draw(main_obj, *everything_else)

# circles -> draw child point
# bars -> draw parent to child, dot on mate

"""
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
mark, = ax.plot([], [], marker='.', markersize=3, color='r')

b3 = PointList(c1, res, rot)
b4 = PointList(c2, res, rot)

c1_points = b3.get_points_list()
c2_points = b4.get_points_list()

c1_x = [i.real for i in c1_points]
c1_y = [i.imag for i in c1_points]

c2_x = [i.real for i in c2_points]
c2_y = [i.imag for i in c2_points]

c1_line, = ax.plot([], [])
c1_mark, = ax.plot([], [], marker='.', markersize=3, color='b')
c2_line, = ax.plot([], [])
c2_mark, = ax.plot([], [], marker='.', markersize=3, color='g')

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
        # self.mate_lengthotation_step_list = [resolution * i for i in range(self.number_of_points)]

        self.ax = pyplot.subplot()
        self.ax.set_aspect('equal')

    def plot(self):
        # points_list = [self.object.get_point(i) for i in self.mate_lengthotation_step_list]
        points_list = self.get_points_list()
        x_list = [i.real for i in points_list]
        y_list = [i.imag for i in points_list]

        # self.ax.plot(x_list, y_list)
        pyplot.plot(x_list, y_list)
"""
