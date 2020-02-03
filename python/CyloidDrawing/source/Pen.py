from Point import Point
from typing import Type, List
from Object import Object
from

class Pen:

    def __init__(self, pa: Type[Object]):
        super().__init__()
        self.dots: List[Point] = []
        self.parent_anchor: Type[Object] = pa

    def get_next_dot(self):
        self.dots.append(self.parent_anchor.get_child_anchor())
    