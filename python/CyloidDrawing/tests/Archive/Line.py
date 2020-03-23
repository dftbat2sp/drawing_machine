from tests.Archive.Object import Object

class Line(Object):
    def __init__(self, pa):
        super(Line, self).__init__(pa)

    def get_child_anchor(self):
        pass