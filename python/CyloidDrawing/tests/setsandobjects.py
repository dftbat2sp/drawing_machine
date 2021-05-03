import collections

class notMoving:
    pass

class moving:
    pass

class parentTreeError(Exception):
    pass

class mateTreeError(Exception):
    pass

class noParentError(Exception):
    pass

class anchor(notMoving):
    def __init__(self):
        self.point = 1
        
    def get_parent_tree(self):
        raise noParentError

class circle(moving):
    def __init__(self, parent):
        self.parent = parent
    
    def get_parent_tree(self):
        parent_tree = [self]
        if issubclass(self.parent.__class__, notMoving):
            parent_tree.append(self.parent)
        elif issubclass(self.parent.__class__, moving):
            parent_tree.extend(self.parent.get_parent_tree())
        else:
            raise parentTreeError
        
        parent_tree = list(dict.fromkeys(parent_tree))
        
        return parent_tree

class bar(moving):
    def __init__(self, parent, mate):
        self.parent = parent
        self.mate = mate
    
    def get_parent_tree(self):
        parent_tree = [self]
        if issubclass( self.parent.__class__, notMoving):
            parent_tree.append(self.parent)    
        elif issubclass(self.parent.__class__, moving):
            parent_tree.extend(self.parent.get_parent_tree())
        else:
            raise parentTreeError
        
        if issubclass(self.mate.__class__, notMoving):
            parent_tree.append(self.mate)
        elif issubclass(self.mate.__class__, moving):
            parent_tree.extend(self.mate.get_parent_tree())
        else:
            raise mateTreeError
        
        parent_tree = list(dict.fromkeys(parent_tree))
        
        return parent_tree
      
        
        
a1 = anchor()
a2 = anchor()
c1 = circle(a1)
c2 = circle(a2)
bc1c2 = bar(c1, c2)

print(bc1c2.get_parent_tree())
