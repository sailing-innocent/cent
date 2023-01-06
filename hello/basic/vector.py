from math import hypot

class Vector:
    def __init__(self, x = 0, y = 0):
        self.x = 0
        self.y = 0
    def __repr__(self):
        # for debugger print()
        # __str__ is only for string, if no __str__, it will use __repr__ instead
        return 'Vector(%r,%r)' % (self.x, self.y)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        # for if(Vector)
        return bool(abs(self))
    
    def __add__(self, other):
        # for + 
        x = self.x + other.x
        y = self.y + other.y

    def __mul__(self, scalar):
        # for * 
        return Vector(self.x * scalar, self.y * scalar)