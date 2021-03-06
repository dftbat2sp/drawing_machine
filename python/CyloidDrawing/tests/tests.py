import numpy

temp = numpy.inf
temp2 = numpy.NINF
other = 5

print(min(other, temp))

'''
print(12.5 % 2.2)
a = -50
a += 360

print(a)
b = 20
b %= 3

print(b)
'''

'''
def return_int():
    return 1

myint: int = 1

myint = return_int()

print(myint)
'''

'''
class Test:
    def __init__(self, i):
        self.item = i

    def get_test(self):
        return self


item1 = Test(1)
item2 = Test(2)

print(item1.get_test().item)
print(item2.get_test().item)
'''

"""from Point import Point
from Circle import Circle
from Angle import Angle
from Angle_Type import Angle_Type

p = Point(1,1)
csa = Angle(90, angle_type=Angle_Type.deg)
c = Circle(p, 1, csa, 1)
print(c.get_child_anchor())"""

"""from sympy import *
from sympy.vector import *
# import numpy as np

init_printing(use_unicode=True)
# length = 5
degrees = 45


theta = rad(degrees)
rm = Matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
# x,y
x = Matrix([3, 4])
x = rm*x


print(x)

from SimpleVector import SimpleVector
# from sympy import pi

v = SimpleVector(3, 4)
print(v.vector)
v.rotate_vector(0.005)
# v.rotate_vector(45, deg=True)
print(v.vector)"""

"""from sympy import *

x = Matrix([1, 1])
y = Matrix([3, 4])
z = Matrix([x.row(0), y.row(1)])

print(x)
print(x+y)
print(x.row(0))
print(z)"""

"""
from SimpleVector import SimpleVector

x = SimpleVector(1,2)
y = SimpleVector(3,4)
z = y + x
print(z.vector)"""

"""
from Angle import Angle
from sympy import Rational

a = Angle(90, deg=True)
b = Angle(Rational(180/2), deg=True)

print(a.magnitude)
print(b.magnitude)
"""

"""
# Resolution Test
from sympy import Rational

resoltuion = Rational(1, 100)
ratio2 = Rational(7, 13)
count = 100000000
angle_start = 0
angle = angle_start
# step = ratio * resoltuion
res = 0.01
ratio = 7/13
step = res * ratio

print(f'count: {count:,}')
print(f'res:   {res}')
print(f'step:  {step}')

# for num in range(count):
#     angle += step
# print(f'for loop:      {angle}')

angle2 = angle_start + (count * ratio * res)

print(f'decimal calc:  {angle2}')

angle3 = angle_start + (count * ratio * resoltuion)

print(f'rational calc: {angle3}')
"""
"""
from Rotor import Rotor
import numpy as np
from time import time

r = Rotor(1, 1, 0)
step = 0.01
stepper = step
x = 1
y = 2
count = 100000


print(f'{count:,}')
print(time()%3600)

for num in range(count):
    r.get_step(0)
    # x_new = x * np.cos(stepper) - y * np.sin(stepper)
    # y_new = x * np.sin(stepper) + y * np.cos(stepper)
    # x = x_new
    # y = y_new
    # stepper += step

print(time()%3600)
"""
"""
import numpy as np

class svec:

    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def rotate_vec(self, theta, deg=False):
        if deg:
            theta = np.radians(theta)

"""
"""
import shapely.geometry as sg
import shapely.affinity as sa
import numpy as np

p = sg.Point(1,1)
# p = p + p
p2 = sa.rotate(p, 0, origin=sg.Point(1,1))

print(p.coords)
print(p2)
"""
"""
from Point import *
from numpy import pi

p = Point(1,0)
q = Point(-1,0)
r = rotate_point(p, pi)

print(rotate_point(p, pi/2, Point(1,1)))
print(rotate_point(q, pi/2, Point(0,0)))
print(r)
"""
"""
import cmath
from cmath import exp, pi

p1 = 1 + 1j
p2 = 4 + 5j

angle = p2 - p1

ang2 = 5 * exp(0.9272952180016122 * 1j1)

print(ang2)
"""
"""
class IntersectionError(Exception):
    pass

import math

def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1 :
        raise IntersectionError(f'Non-intersecting, non-concentric circles not contained within each other.')
    # One circle within other
    if d < abs(r0-r1):
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

        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d
        y2=y0+a*(y1-y0)/d
        x3=x2+h*(y1-y0)/d
        y3=y2-h*(x1-x0)/d

        return x3 + y3 * 1j

cmplx = get_intersections(0, -5, 10, 0, 5, 10) # vertical points, + / -, right bias
print(f'first intersection:  {cmplx}')
cmplx = get_intersections(0, 5, 10, 0, -5, 10) # vertical points, -/+, left bias
print(f'first intersection:  {cmplx}')
cmplx = get_intersections(-5, 0, 10, 5, 0, 10) # horizontal points, -/+ bottom bias
print(f'first intersection:  {cmplx}')
cmplx = get_intersections(5, 0, 10, -5, 0, 10) # horizontal points, +/- top bias
print(f'first intersection:  {cmplx}')
cmplx = get_intersections(5, 5, 10, -5, -5, 10) # horizontal points, +/- top bias
print(f'first intersection:  {cmplx}')
cmplx = get_intersections(3, -5, 4, -3, -5, 4) # horizontal points, +/- top bias
print(f'first intersection:  {cmplx}')

# x1 > x2 = top bias
# y1 > y2 = left bias

# want x0 > x1
# want y1 > y0

"""
"""
from typing import Type, Tuple
mate1 = 2 + 2j
mate2 = 3 - 3j, 3.2437

print(isinstance(mate2, tuple))
print(len(mate1))
"""
"""
import numpy as np

resolution = 0.0001
num_of_rotations = 0.5
rotations_radians = num_of_rotations * np.pi * 2
# TODO
num_of_points = int(np.ceil(num_of_rotations / resolution))

point_list = np.linspace(1, rotations_radians, num_of_points, dtype=float)

exp_const = 2 * np.pi * 1j
rotn_speed = 2
length = 5
start_angle = 0
length_start_angle_const = length * np.exp(start_angle * 1j)

modified_points = point_list + (length_start_angle_const * np.exp(exp_const * rotn_speed * point_list))

# unit_vector = pow(modified_points, 0) * (4 + 3j)
unit_vector = np.full_like(modified_points, 4+3j)

none_list = None
try:
    none_plus_list = none_list + modified_points
except TypeError:
    none_list = np.full_like(modified_points, 1+1j)
    none_plus_list = none_list + modified_points


points1 = np.array([1+1j, 2+1j, 3+1j, 4+1j, 5+1j])
points2 = np.array([7+1j, 8+1j, 9+1j, 11+1j, 11+1j])

radius1 = 5
radius2 = 5

d = np.sqrt(np.power(points2.real - points1.real, 2) + np.power(points2.imag - points1.imag, 2))

# WHAT!?!
if (d > 6).any():
    print("what")

if (d < abs(10 - 3)).any():
    print("huh?")

if (d == 0).any():
    print("ya")

# print(f'num of points: {num_of_points} (len: {len(point_list)})')
# print(point_list[:12])
# print(modified_points[:12])
# print(unit_vector[:12])

print(points1)
print(points2)
print(d)
"""
"""
from numpy import ndarray, sqrt, power, array
import numpy as np
from typing import Type
import matplotlib.pyplot as plt

class IntersectionError(Exception):
    pass


class CircleMate:

    def __init__(self, radius: float, centers: ndarray):
        self.r: float = radius
        self.c: ndarray[complex] = centers
        self.points: Type[ndarray[complex], None] = None

    def get_circles_intersections(self, mate) -> None:
        # circle 1: (x0, y0), radius r0
        # circle 2: (x1, y1), radius r1
        # x1 = mate
        # x0 = self
        # d = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        d = sqrt(power((mate.c.real - self.c.real), 2) + power((mate.c.imag - self.c.imag), 2))

        # non intersecting
        if (d > (mate.r + self.r)).any():
            raise IntersectionError(f'Non-intersecting, non-concentric circles not contained within each other.')
        # One circle within other
        elif (d < abs(mate.r - self.r)).any():
            raise IntersectionError(f'Non-intersecting circles. One contained in the other.')
        # coincident circles
        elif (d == 0).any():
            raise IntersectionError(f'Concentric circles.')
        else:
            # to ensure a bias to top and right
            # if x0 < x1:
            #     x0, x1 = x1, x0
            #     r0, r1 = r1, r0
            # if y0 > y1:
            #     y0, y1 = y1, y0
            #     r0, r1 = r1, r0
            #
            # a = (self.r ** 2 - mate.r ** 2 + d ** 2) / (2 * d)
            # h = sqrt(self.r ** 2 - ((self.r ** 2 - mate.r ** 2 + d ** 2) / (2 * d)) ** 2)
            # x2 = self.c.real + a * (mate.c.real - self.c.real) / d
            # y2 = self.c.imag + a * (mate.c.imag - mate.c.imag) / d
            # x3 = x2 + h * (y1 - y0) / d
            # x3 = (self.c.real + a * (mate.c.real - self.c.real) / d) + h * (mate.c.imag - self.c.imag) / d
            # y3 = y2 - h * (x1 - x0) / d
            # y3 = (self.c.imag + a * (mate.c.imag - mate.c.imag) / d) - h * (mate.c.real - self.c.real) / d

            # self.points = np.where()

            self.points = \
                ((self.c.real + ((self.r ** 2 - mate.r ** 2 + d ** 2) / (2 * d)) * (mate.c.real - self.c.real) / d) + (
                    sqrt(self.r ** 2 - ((self.r ** 2 - mate.r ** 2 + d ** 2) / (2 * d)) ** 2)) * (
                         mate.c.imag - self.c.imag) / d) + \
                ((self.c.imag + ((self.r ** 2 - mate.r ** 2 + d ** 2) / (2 * d)) * (mate.c.imag - mate.c.imag) / d) - (
                    sqrt(self.r ** 2 - ((self.r ** 2 - mate.r ** 2 + d ** 2) / (2 * d)) ** 2)) * (
                         mate.c.real - self.c.real) / d) * 1j

def get_intercetions(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d
        y2=y0+a*(y1-y0)/d
        x3=x2+h*(y1-y0)/d
        y3=y2-h*(x1-x0)/d

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        return x3, y3, x4, y4


# c1 = CircleMate(5, array([-2.5 + 1j, -2 + 1j, -1 + 1j, -0.5 + 1j, 1 + 1j,  2 + 1j]))
# c2 = CircleMate(5, array([2.5 + 1j,   2 + 1j,  1 + 1j,  0 + 1j,  -1 + 1j, -2 + 1j]))
# c1 = CircleMate(5, array([-2 + 1j,  2 + 1j]))
# c2 = CircleMate(5, array([ 2 + 1j, -2 + 1j]))
c1 = CircleMate(4, array([ 2 + 1.5j, -2 - 1.5j]))
c2 = CircleMate(4, array([-2 - 1.5j,  2 + 1.5j]))

c1.get_circles_intersections(c2)


centercir = plt.Circle((0,0), 1.9, color='linen')
movecir1 = plt.Circle((2,2), 1, color='silver')
movecir2 = plt.Circle((-2,2), 1, color='silver')
movecir3 = plt.Circle((-2,-2), 1, color='silver')
movecir4 = plt.Circle((2,-2), 1, color='silver')
# print(c1.points)
p2m2 = np.linspace(2, -2)
m2p2 = np.linspace(-2, 2)

xpyp2xmyp = [get_intercetions(i, 2, 1, 0, 0, 1.9) for i in p2m2]
xmyp2xmym = [get_intercetions(-2, i, 1, 0,0,1.9) for i in p2m2]
xmym2xpym = [get_intercetions(i, -2, 1, 0,0,1.9) for i in m2p2]
xpym2xpyp = [get_intercetions(2, i, 1, 0,0,1.9) for i in m2p2]

# x+y+ -> x-y+
# for i in range(len(x)):
    # print(f'x: {i:22} = {get_intercetions(i, 2, 1, 0, 0, 1.9)}')
    # print(f'x: {x[i]:22} = {plusplus[i]}')


fig = plt.figure(figsize=(10,10))
ax = plt.axes(xlim=(-3,3), ylim=(-3,3))

interx1 = [i[0] for i in xpyp2xmyp]
intery1 = [i[1] for i in xpyp2xmyp]

interx2 = [i[0] for i in xmyp2xmym]
intery2 = [i[1] for i in xmyp2xmym]

interx3 = [i[0] for i in xmym2xpym]
intery3 = [i[1] for i in xmym2xpym]

interx4 = [i[0] for i in xpym2xpyp]
intery4 = [i[1] for i in xpym2xpyp]

# print(plusplus)
# print(interx)


ax.add_artist(centercir)
ax.add_artist(movecir1)
ax.add_artist(movecir2)
ax.add_artist(movecir3)
ax.add_artist(movecir4)
ax.plot(interx1, intery1, color='c')
ax.plot(interx2, intery2, color='dodgerblue')
ax.plot(interx3, intery3, color='deeppink')
ax.plot(interx4, intery4, color='lawngreen')

plt.show()

# for i in x:
    # print(f'x: {i:22} = {get_intercetions(i, 2, 1, 0, 0, 1.9)}')

# print(get_intercetions(2,1.5,4,-2,-1.5,3))
# print(get_intercetions(-2,-1.5,3,2,1.5,4))
# print(get_intercetions(2,-1.5,3,-2,1.5,4))
# print(get_intercetions(-2,1.5,3,2,-1.5,4))
"""
"""
import itertools
import numpy as np
l1 = np.array([1 + 1j,2 + 1j,3 + 1j,4 + 1j,5 + 3j])
l2 = np.array([2 + 1j,3 + 1j,4 + 1j,5 + 4j,6 + 1j])

point_list_real = [l1.real, l2.real]
point_list_imag = [l1.imag, l2.imag]

print(max(itertools.chain.from_iterable(point_list_real)))
print(max(itertools.chain.from_iterable(point_list_imag)))
# print(max(l1, l2))
"""
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

widths = [2, 3, 1.5]
heights = [1, 3, 2]

gs_kw = dict(width_ratios=widths, height_ratios=heights)
fig6, f6_axes = plt.subplots(ncols=3, nrows=3, constrained_layout=True,
                             gridspec_kw=gs_kw)

fig6.set_size_inches(10,10)
for r, row in enumerate(f6_axes):
    for c, ax in enumerate(row):
        label = 'Width: {}\nHeight: {}'.format(widths[c], heights[r])
        ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
        ax.set_aspect('equal')

plt.show()
"""
"""
import numpy as np

parent_point_array = np.array([1 + 0j])
mate_point_array = np.array([-1 + 0j])

point_length = 1
arm_length = 1
arm_angle = -np.pi / 2

print(np.angle(mate_point_array - parent_point_array) )

bar_point_array = parent_point_array + (point_length * np.exp(np.angle(mate_point_array - parent_point_array) * 1j)) + (
        arm_length * np.exp((np.angle(mate_point_array - parent_point_array) + arm_angle) * 1j))

print(bar_point_array)

# bar_arm_point_array = bar_point_array + (arm_length * np.exp(arm_angle * 1j))

# print(bar_arm_point_array)
"""