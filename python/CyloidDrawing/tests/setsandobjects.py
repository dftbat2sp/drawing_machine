import collections

class notMoving:
    pass

class moving:
    pass

class parentTreeError(Exception):
    pass

class mateTreeError(Exception):
    pass

class anchor(notMoving):
    def __init__(self):
        self.point = 1

class circle(moving):
    def __init__(self, parent):
        self.parent = parent
    
    def get_parent_tree(self):
        if issubclass(self.parent, notMoving):
            parent_tree = { self.parent: None}
        elif issubclass(self.parent, moving):
            parent_tree = { self.parent: None, **self.parent.get_parent_tree()}
        else:
            raise parentTreeError
        
        return parent_tree

class bar(moving):
    def __init__(self, parent, mate):
        self.parent = parent
        self.mate = mate
    
    def get_parent_tree(self):
        if issubclass(self.parent, notMoving):
            parent_tree = { self.parent: None}
        
        
        
        if issubclass(self.parent, notMoving):
            parent_tree = { self.parent: None,}
        elif issubclass(self.parent, moving):
            parent_tree = { self.parent: None, **self.parent.get_parent_tree()}
        else:
            raise parentTreeError
        
        return parent_tree
      
        
        
a11 = anchor()
c11 = circle(a11)
c12 = 
