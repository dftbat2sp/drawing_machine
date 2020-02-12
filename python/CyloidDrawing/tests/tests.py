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

from Point import *
from numpy import pi

p = Point(1,0)
q = Point(-1,0)
r = rotate_point(p, pi)

print(rotate_point(p, pi/2, Point(1,1)))
print(rotate_point(q, pi/2, Point(0,0)))
print(r)











