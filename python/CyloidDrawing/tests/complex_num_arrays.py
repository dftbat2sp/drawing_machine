import numpy as np
import math

# print(1/np.sinh(1))

# a = 1 + 1j


nparray = np.linspace(0,7,8)
print(nparray)
nparray = nparray + (2+1j)
print(nparray)
nparray = nparray * 2
print(nparray)
nparray[:].real = nparray[:].real * 2
print(nparray)
nparray[:].imag = nparray[:].imag + 2
print(nparray)
nparray = nparray[:].real * 2 + nparray[:].imag * 3
print(nparray)

