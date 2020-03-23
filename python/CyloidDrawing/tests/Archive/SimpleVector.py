from sympy import Matrix, rad, symbols, cos, sin
from typing import Union


class SimpleVector:
    t = symbols('t')
    rotation_matrix = Matrix([[cos(t), -sin(t)], [sin(t), cos(t)]])

    def __init__(self, x, y):
        self.vector = Matrix([x, y])
        self.vector.simplify()


    def rotate_vector(self, theta, deg: bool = False):
        # def rotate_vector(self, theta: Union[int, float], deg: bool = False):
        if deg:
            theta = rad(theta)

        self.vector = self.rotation_matrix.subs(self.t, theta) * self.vector
        # self.vector.simplify()

    # def rotate_new_vector(self, theta, deg: bool = False):
    #     if deg:
    #         theta = rad(theta)
    #
    #     new_vector = SimpleVector(self.vector.row(0), self.vector.row(1))
    #     new_vector.rotate_vector(theta)
    #     # new_vector.vector.simplify()
    #
    #     return new_vector


    def get_step(self, step_count):
        return self

    def __add__(self, other):
        new_vector: Matrix = self.vector + other.vector
        new_vector.simplify()
        # return SimpleVector(self.vector.row(0), self.vector.row(1))
        return SimpleVector(new_vector.row(0), new_vector.row(1))
