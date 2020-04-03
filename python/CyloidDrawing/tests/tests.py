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

import numpy as np

resolution = 0.0001
num_of_rotations = 4000
rotations_radians = num_of_rotations * np.pi * 2
# TODO
num_of_points = int(np.ceil(num_of_rotations / resolution))

point_list = np.linspace(1, rotations_radians, num_of_points, dtype=float)

exp_const = 2 * np.pi * 1j
rotn_speed = 2
length = 5
start_angle = 0
length_start_angle_const = length * np.exp(start_angle * 1j)

modified_points = length_start_angle_const * np.exp(exp_const * rotn_speed * point_list)

print(f'num of points: {num_of_points} (len: {len(point_list)})')

print(point_list[:12])
print(modified_points[:12])

import time
time.sleep(10)
