import copy
class A:
  def __init__(self):
    self.t = 0.
a = A()
b = copy.deepcopy(a)
b.t = 3
print(hex(id(a)))
print(hex(id(b)))
a.t = copy.deepcopy(b.t)
print(a.t)
print(hex(id(a.t)))
print(hex(id(b.t)))