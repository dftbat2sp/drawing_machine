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

    def get_min_max_values(self) -> Tuple:
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

    def get_min_max_values_normalized_to_origin(self) -> Tuple:
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
        self.mate_point_array: Type[np.ndarray, None] = None

    def get_circle_intersection_with_mate(self, base_points_list: RotationResolution) -> None:
        raise NotImplementedError

    def get_parent_points(self, base_points_list: RotationResolution) -> None:
        raise NotImplementedError


class Anchor(Anchorable):

    def __init__(self, complex_number: complex):
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

    def get_min_max_values(self) -> Tuple:
        return self.point, self.point, self.point, self.point

    def get_min_max_values_normalized_to_origin(self) -> Tuple:
        return 0, 0, 0, 0

    def __add__(self, other):
        return Anchor(self.point + other.point)


class Circle(Anchorable, BarMateSlide):
    exp_const = math.tau * 1j

    def __init__(self, radius: float, frequency: float, starting_angle: float = 0, deg: bool = False,
                 parent_object: Anchorable = Anchor(0 + 0j)):
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

        self.parent: Anchorable = parent_object

        """
        Drawing Objects
        """
        self.main_circle_edge_artist = plt.Circle((0, 0), self.radius, fill=False)
        self.main_centre2point_line_artist = plt.Line2D([], [], marker='o', markevery=(1, 1))

        self.secondary_circle_edge_artist = plt.Circle((0, 0), self.radius, fill=False)
        self.secondary_centre2point_line_artist = plt.Line2D([], [], marker='o', markevery=(1, 1))

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

    def get_min_max_values(self) -> Tuple:
        x_min = min(self.point_array[:].real)
        x_max = max(self.point_array[:].real)

        y_min = min(self.point_array[:].imag)
        y_max = max(self.point_array[:].imag)

        return x_min, x_max, y_min, y_max

    def get_min_max_values_normalized_to_origin(self) -> Tuple:
        x_max = self.radius/2
        x_min = -1 * x_max

        y_max = x_max
        y_min = x_min

        return x_min, x_max, y_min, y_max

    def __str__(self):
        return f'radius: {self.radius}, freq: {self.rotation_frequency}, angle: {self.starting_angle}'


class Bar(Anchorable, BarMateFix):

    def __init__(self, parent_object, point_length_from_parent, mate_object, mate_length_from_parent):
        super().__init__()
        self.parent: Type[Anchorable] = parent_object
        self.mate: Type[Anchorable, BarMateSlide, BarMateFix] = mate_object
        self.point_length = point_length_from_parent
        self.mate_length = mate_length_from_parent

        """
        Drawing Objects
        """
        self.main_mate_line = plt.Line2D([], [], marker='D', markevery=(1, 1), linestyle='--')
        self.main_point_line = plt.Line2D([], [], marker='o', markevery=(1, 1))

        self.secondary_mate_line = plt.Line2D([], [], marker='D', markevery=(1, 1), linestyle='--')
        self.secondary_point_line = plt.Line2D([], [], marker='o', markevery=(1, 1))

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
                self.mate_point_array = self.parent.point_array

            self.point_array = self.parent.point_array + (
                    self.point_length * math.exp(cmath.phase(self.mate_point_array - self.parent.point_array) * 1j))

    def update_drawing_objects(self, frame) -> None:
        # MAIN
        self.main_mate_line.set_data(
            [self.parent.point_array[frame].real, self.mate_point_array[frame].real],
            [self.parent.point_array[frame].imag, self.mate_point_array[frame].imag]
        )
        self.main_point_line.set_data(
            [self.parent.point_array[frame].real, self.point_array[frame].real],
            [self.parent.point_array[frame].imag, self.point_array[frame].imag]
        )

        # SECONDARY
        self.secondary_mate_line.set_data(
            [0, self.mate_point_array[frame].real - self.parent.point_array[frame].real],
            [0, self.mate_point_array[frame].imag - self.parent.point_array[frame].imag]
        )
        self.secondary_point_line.set_data(
            [0, self.point_array[frame].real - self.parent.point_array[frame].real],
            [0, self.point_array[frame].imag - self.parent.point_array[frame].imag]
        )

    def get_main_drawing_objects(self) -> List:
        return [self.main_point_line, self.main_mate_line]

    def get_secondary_drawing_objects(self) -> List:
        return [self.secondary_point_line, self.secondary_mate_line]

    def get_min_max_values(self) -> Tuple:
        x_min = min(itertools.chain(self.point_array[:].real, self.mate_point_array[:].real))
        x_max = max(itertools.chain(self.point_array[:].real, self.mate_point_array[:].real))

        y_min = min(itertools.chain(self.point_array[:].imag, self.mate_point_array[:].imag))
        y_max = max(itertools.chain(self.point_array[:].imag, self.mate_point_array[:].imag))

        return x_min, x_max, y_min, y_max

    def get_min_max_values_normalized_to_origin(self) -> Tuple:
        x_min = min(itertools.chain(
            self.point_array[:].real - self.parent.point_array[:].real,
            self.mate_point_array[:].real  - self.parent.point_array[:].real
        ))
        x_max = max(itertools.chain(
            self.point_array[:].real  - self.parent.point_array[:].real,
            self.mate_point_array[:].real - self.parent.point_array[:].real
        ))

        y_min = min(itertools.chain(
            self.point_array[:].imag  - self.parent.point_array[:].imag,
            self.mate_point_array[:].imag  - self.parent.point_array[:].imag
        ))
        y_max = max(itertools.chain(
            self.point_array[:].imag  - self.parent.point_array[:].imag,
            self.mate_point_array[:].imag  - self.parent.point_array[:].imag
        ))

        return x_min, x_max, y_min, y_max

    def get_circle_intersection_with_mate(self, base_points_list: RotationResolution) -> None:
        if self.mate_point_array is None:
            self.mate.get_parent_points(base_points_list)

            d: np.ndarray = np.sqrt(
                np.power((self.parent.point_array.real - self.mate.parent.point_array.real), 2) +
                np.sqrt(np.power((self.parent.point_array.imag - self.mate.parent.point_array.imag), 2)))

            if (d > (self.mate_length + self.mate.mate_length)).any():
                raise IntersectionError(f'Non-intersecting, non-concentric circles not contained within each other.')
            elif (d < abs(self.mate_length - self.mate.mate_length)).any():
                raise IntersectionError(f'Non-intersecting circles. One contained in the other.')
            elif (d == 0).any():
                raise IntersectionError(f'Concentric circles.')
            else:
                """
                # doing the below calc in fewer varialbes to save on memory
                a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
                h = sqrt(r0 ** 2 - a ** 2)
                x2 = x0 + a * (x1 - x0) / d
                y2 = y0 + a * (y1 - y0) / d
                x3 = x2 + h * (y1 - y0) / d
                y3 = y2 - h * (x1 - x0) / d
                """
                # these variables kept out of larger equation for ease of reading and deduplicate processing the equation multiple times
                a = (np.power(self.mate_length, 2) - np.power(self.mate.mate_length, 2) + np.power(d, 2)) / (2 * d)
                h = np.sqrt(np.power(self.mate_length, 2) - np.power(a, 2))

                # @formatter:off
                self.mate_point_array = (
                    (self.parent.point_array.real + (a * (
                            self.mate.parent.point_array.real - self.parent.point_array.real) / d)) + (
                        h * (self.mate.parent.point_array.imag - self.parent.point_array.imag) / d)
                        ) + (1j * (
                    (self.parent.point_array.imag + (a * (
                            self.mate.parent.point_array.imag - self.parent.point_array.imag) / d)) - (
                        h * (self.mate.parent.point_array.real - self.parent.point_array.real) / d)))
                # @formatter:on

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

    num_of_component_plot_rows = 4
    num_of_component_plot_cols = int(math.ceil(len(components) / num_of_component_plot_rows))

    print(f'plot component_col: {num_of_component_plot_cols}')

    gs_component_plots = gridspec.GridSpecFromSubplotSpec(num_of_component_plot_rows, num_of_component_plot_cols,
                                                     subplot_spec=gs_parent[0])

    gs_main_plot_num_of_rows = 1
    gs_main_plot_num_of_cols = 2
    gs_main_plots = gridspec.GridSpecFromSubplotSpec(gs_main_plot_num_of_rows, gs_main_plot_num_of_cols, subplot_spec=gs_parent[1])

    main_animated_draw_axis = fig.add_subplot(gs_main_plots[:, 0])
    main_final_shape_axis = fig.add_subplot(gs_main_plots[:, 1])

    """ min/max for main drawer axes """
    drawer_x_min, drawer_x_max, drawer_y_min, drawer_y_max = drawer.get_min_max_values()
    axis_space_buffer = 1

    # set xy limits for final drawing axes
    main_animated_draw_axis.set_xlim((drawer_x_min - axis_space_buffer, drawer_x_max + axis_space_buffer))
    main_animated_draw_axis.set_ylim((drawer_y_min - axis_space_buffer, drawer_y_max + axis_space_buffer))

    main_final_shape_axis.set_xlim((drawer_x_min - axis_space_buffer, drawer_x_max + axis_space_buffer))
    main_final_shape_axis.set_ylim((drawer_y_min - axis_space_buffer, drawer_y_max + axis_space_buffer))

    # set axis aspect ratio for final drawing axes
    main_animated_draw_axis.set_aspect('equal')
    main_final_shape_axis.set_aspect('equal')

    main_animated_drawer_final_line, = main_animated_draw_axis.plot([], [])
    # finished_final_line, = main_final_shape_axis.plot(drawer.point_array.real, drawer.point_array.imag)
    main_final_shape_axis.plot(drawer.point_array.real, drawer.point_array.imag)

    comp_axis_list = []
    comp_axis_artist_list = []
    comp_axis_artist_list_list = []

    for num, comp in enumerate(components):
        component_col = int(math.floor(num / num_of_component_plot_rows))
        component_row = num % num_of_component_plot_rows

        component_x_min = min(comp.point_array.real)
        component_x_max = max(comp.point_array.real)
        component_y_min = min(comp.point_array.imag)
        component_y_max = max(comp.point_array.imag)

        # circ_max = (comp.radius / 2) + 0.6
        # circ_min = -1 * circ_max

        temp_ax = fig.add_subplot(gs_component_plots[component_row, component_col])
        temp_ax.set_xlim((component_x_min, component_x_max))
        temp_ax.set_ylim((component_y_min, component_y_max))
        temp_ax.set_aspect('equal')


        # TODO do I need all these lists?
        # Does all this stuff need tobe in INIT?
        comp_axis_list.append(temp_ax)

        comp_axis_artist_list_list.append(comp.get_secondary_drawing_objects())
        comp_axis_artist_list.extend(comp.get_secondary_drawing_objects())

    # individual components

    def init():
        for artist in itertools.chain(main_plot_drawer_artist_list, main_plot_component_artist_list):
            main_animated_draw_axis.add_artist(artist)

        for i, artist_list in enumerate(comp_axis_artist_list_list):
            for artist in artist_list:
                comp_axis_list[i].add_artist(artist)

        return itertools.chain([main_animated_drawer_final_line], main_plot_drawer_artist_list, main_plot_component_artist_list,
                               comp_axis_artist_list)
        # return itertools.chain([main_animated_drawer_final_line], main_plot_drawer_artist_list, main_plot_component_artist_list)
        # return [main_animated_drawer_final_line]

    def get_frames():
        for i in range(drawer.point_array.size):
            point = i * 20
            if point < drawer.point_array.size:
                yield point

    def animate(frame):

        drawer.update_drawing_objects(frame)

        for anchorable_obj in components:
            anchorable_obj.update_drawing_objects(frame)
        # frame = frame - 1
        main_animated_drawer_final_line.set_data(drawer.point_array[:frame + 1].real, drawer.point_array[:frame + 1].imag)

        # for axis in comp_axis_list:
        # axis.set
        # drawer_marker.set_data(drawer.point_array[frame].real, drawer.point_array[frame].imag)
        # dl1.set_data([drawer.parent.point_array[frame].real, drawer.point_array[frame].real],
        #              [drawer.parent.point_array[frame].imag, drawer.point_array[frame].imag])
        # # print(drawer.point_array[:frame].real)
        #
        # c1.set_center((supports[2].parent.point_array[frame].real, supports[2].parent.point_array[frame].imag))
        # l1.set_data([supports[2].parent.point_array[frame].real, supports[2].point_array[frame].real],
        #             [supports[2].parent.point_array[frame].imag, supports[2].point_array[frame].imag])

        return itertools.chain([main_animated_drawer_final_line], main_plot_drawer_artist_list, main_plot_component_artist_list,
                               comp_axis_artist_list)
        # return itertools.chain([main_animated_drawer_final_line], main_plot_drawer_artist_list, main_plot_component_artist_list)
        # return [main_animated_drawer_final_line]

    mpl_animation.FuncAnimation(fig, animate,
                                init_func=init,
                                interval=1,
                                blit=True,
                                save_count=1,
                                frames=get_frames,
                                repeat=False)

    plt.show()


anchor1 = Anchor(1 + 1j)
circle1 = Circle(1, 0.233, parent_object=anchor1)
circle2 = Circle(0.35, 0.377, parent_object=circle1)
circle3 = Circle(0.1, 2, parent_object=circle2)
circle4 = Circle(0.2, 0.01, parent_object=circle3)
circle5 = Circle(0.2, 0.01, starting_angle=1, parent_object=circle4)
base_points = RotationResolution(rotations=5)
# anchor1.create_point_lists(pointlist)

# animate_all(circle1, base_points, anchor1)
# animate_all(circle2, base_points, anchor1, circle1)
# animate_all(circle3, base_points, anchor1, circle1, circle2)
# animate_all(circle4, base_points, anchor1, circle1, circle2, circle3)
# animate_all(circle5, base_points, circle1, circle2, circle3, circle4)
animate_all(circle5, base_points, circle1, circle2, circle3)
"""
x2 = x0 + a * (x1 - x0) / d
y2 = y0 + a * (y1 - y0) / d

x2 = (self.parent.point_array.real + (a * (self.mate.parent.point_array.real - self.parent.point_array.real) / d))
y2 = (self.parent.point_array.imag + (a * (self.mate.parent.point_array.imag - self.parent.point_array.imag) / d))
x2 = self.parent.point_array.real + (a * (self.mate.parent.point_array.real - self.parent.point_array.real) / d)
y2 = self.parent.point_array.imag + (a * (self.mate.parent.point_array.imag - self.parent.point_array.imag) / d)

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
