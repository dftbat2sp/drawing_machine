from dataclasses import dataclass
from Anchored import Anchored

@dataclass
class Point(Anchored):
    x: float
    y: float

    def get_child_anchor(self):
        return self
