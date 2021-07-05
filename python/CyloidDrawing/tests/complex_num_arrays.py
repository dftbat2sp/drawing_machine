from multipledispatch.dispatcher import Dispatcher
import numpy as np
import math
from typing import Iterable, List, Tuple, Type, Union, Dict
# from matplotlib import lines
from multipledispatch import dispatch

a = 1

class num():

    def __init__(self, a, b=2) -> None:
        self.a = a
        self.b = b

    def print(self):
        print(f'a: {self.a}, b: {self.b}')

    # __add__ = Dispatcher('__add__')

    # @dispatch(object, (float, int))
    @dispatch((int, float))
    def __add__(self, other):
        return self.a + other

    @dispatch((int, float))
    def __radd__(self, other):
        return self.__add__(other)

    @dispatch(object)
    def __add__(self, other):
        return self.b + other.b



z = num(1)
x = num(3)

z.print()
x.print()

print(f'z+1:   {z + 1}')
print(f'z+1.1: {z + 1.1}')
print(f'z+x:   {z + x}')
print(f'5+z:   {5 + z}')


# z.print()
# x.print()
    

    

# print(a[2])

# print(1/np.sinh(1))

# a = 1 + 1j

# a = [1,2,3,4,5,6,7]

# print(a[1:])

# nparray = np.linspace(0,7,8)
# print(nparray)
# nparray = nparray + (2+1j)
# print(nparray)
# nparray = nparray * 2
# print(nparray)
# nparray[:].real = nparray[:].real * 2
# print(nparray)
# nparray[:].imag = nparray[:].imag + 2
# print(nparray)
# nparray = nparray[:].real * 2 + nparray[:].imag * 3
# print(nparray)

