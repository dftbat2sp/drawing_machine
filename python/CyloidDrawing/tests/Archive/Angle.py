from dataclasses import dataclass
from numpy import pi
from math import radians
from tests.Archive.Misc import sigfig


@dataclass
class Angle:
    magnitude: float
    deg: bool = False
    significant_digits: int = 5

    def __post_init__(self):
        # if self.angle_type == Angle_Type.deg:
        #     self._normalize_degrees()
        # elif self.angle_type == Angle_Type.rad:
        if self.deg:
            self.magnitude = radians(self.magnitude)
        self._normalize_radians()

    # def mag2deg(self):
    #     if self.angle_type == Angle_Type.deg:
    #         return self.magnitude
    #     elif self.angle_type == Angle_Type.rad:
    #         return degrees(self.magnitude)

    # def mag2rad(self):
    #     if self.angle_type == Angle_Type.rad:
    #         return self.magnitude
    #     elif self.angle_type == Angle_Type.deg:
    #         return radians(self.magnitude)

    # def _normalize_degrees(self):
    #     # change degree angles to <=  360
    #     self.angle_type %= 360
    #     # change negative angles to positive angles
    #     if self.magnitude < 0:
    #         self.magnitude += 360

    def _normalize_radians(self):
        # change radian angles to <= 2*pi
        self.magnitude %= 2 * pi
        # change negative angles to positive angles
        if self.magnitude < 0:
            self.magnitude += 2 * pi

    # +
    def __add__(self, other):
        new_angle = self.magnitude + other.magnitude
        return Angle(new_angle)

    # -
    def __sub__(self, other):
        new_angle = self.magnitude - other.magnitude
        return Angle(new_angle)
    # *
    def __mul__(self, other):
        new_angle = self.magnitude * other.magnitude
        return Angle(new_angle)

    # /
    def __truediv__(self, other):
        new_angle = self.magnitude / other.magnitude
        return Angle(new_angle)

    # <
    def __lt__(self, other):
        return sigfig(self.magnitude, self.significant_digits) < sigfig(other.magnitude, self.significant_digits)

    # <=
    def __le__(self, other):
        return sigfig(self.magnitude, self.significant_digits) <= sigfig(other.magnitude, self.significant_digits)

    # >
    def __gt__(self, other):
        return sigfig(self.magnitude, self.significant_digits) > sigfig(other.magnitude, self.significant_digits)

    # >=
    def __ge__(self, other):
        return sigfig(self.magnitude, self.significant_digits) >= sigfig(other.magnitude, self.significant_digits)

    # ==
    def __eq__(self, other):
        return sigfig(self.magnitude, self.significant_digits) == sigfig(other.magnitude, self.significant_digits)
