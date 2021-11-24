import numpy as np
import math

class Int(int):
    def __new__(cls, value):
        return super().__new__(cls, value)

    def __init__(self, value):
        self.num = value

    def __getitem__(self, index):
        return self.num * np.power(math.tau, 1)

x = Int(4)
print(x[6])
