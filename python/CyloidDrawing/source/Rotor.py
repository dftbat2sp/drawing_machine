from abc import ABC

# from sympy import Rational, rad, Point2D
import matplotlib.pyplot as pyplt

import numpy as np
from Point import Point, rotate_point

from dataclasses import dataclass
from typing import Union


class Rotor:

    def __init__(self, length, rpm, angle=0, resolution=0.1, deg: bool = False, parent_object=Point(0, 0)):

        # universal
        self.parent_object = parent_object
        self.length = length
        self.rpm = rpm
        self.rpm_step_ratio = rpm
        if deg:
            self.starting_angle = np.radians(angle)
        else:
            self.starting_angle = angle

        # universal
        self.resolution = resolution

        self.starting_point = rotate_point(Point(length, 0), angle)

    def get_step(self, step_count: int):
        return rotate_point(self.starting_point, step_count * self.rpm_step_ratio * self.resolution,
                            parent=self.parent_object.get_step(step_count))


resolution = 0.01

anchor = Point(0, 0)
# anchor = Point(1, 1)
# c1 = Rotor(2, 1, resolution=resolution, parent_object=anchor)
c1 = Rotor(2, 1, resolution=resolution, parent_object=anchor)
# c2 = Rotor(1, 2, resolution=resolution, parent_object=c1)
c2 = Rotor(1, 5, resolution=resolution, angle=np.pi, parent_object=c1)
# c2 = Rotor(1, 1, resolution=resolution, angle=np.pi, parent_object=c1)
c3 = Rotor(0.5, 11, resolution=resolution, parent_object=c2)

print(f'starting c1: {c1.starting_point}')
print(f'starting c2: {c2.starting_point}')

t = int(np.round(2 * np.pi / resolution))
# t = np.linspace(0, np.pi, num=300)
# c2xpoints = [c2.get_step(i).x for i in range(t)]
# c2ypoints = [c2.get_step(i).y for i in range(t)]

c1xpoints = [c1.get_step(i).x for i in range(t)]
c1ypoints = [c1.get_step(i).y for i in range(t)]

c2xpoints = [c2.get_step(i).x for i in range(t)]
c2ypoints = [c2.get_step(i).y for i in range(t)]

c3xpoints = [c3.get_step(i).x for i in range(t)]
c3ypoints = [c3.get_step(i).y for i in range(t)]
#
# for i in range(len(c2xpoints)):
#     print(f'x1: {c1xpoints[i]}, y1: {c1ypoints[i]}')
#     print(f'x2: {c2xpoints[i]}, y2: {c2ypoints[i]}')

fig1, ax1 = pyplt.subplots()

ax1.set_aspect('equal')

# ax1.plot(c1xpoints, c1ypoints)
# ax1.plot(c2xpoints, c2ypoints)
ax1.plot(c3xpoints, c3ypoints)
# pyplt.axis([-5, 5, -5, 5])
# pyplt.a
pyplt.show()
