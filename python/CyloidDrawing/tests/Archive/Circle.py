from tests.Archive.Angle import Angle
from tests.Archive.Object import Object

class Circle(Object):
    def __init__(self, pa, rpm, starting_angle, radius):
        super().__init__(pa)
        # + = CW
        # - = CCW
        self.rpm: float = rpm
        self.angle: Angle = starting_angle
        self.radius = radius

    def get_child_anchor(self):
        # calculate parent anchor
        self.parent_anchor.get_step()

        # calculate child anchor from parent anchor plus angle change
        self.child_anchor = self.new_point_from_angle_and_length(self.parent_anchor, self.angle, self.radius)

        return self.child_anchor




