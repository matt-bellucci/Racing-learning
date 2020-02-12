import copy
import random
class A:
	def __init__(self):
		self.attr = 0.

# copy.deepcopy
print("=== copy.deepcopy ===")
print()
a = A()
b = copy.deepcopy(a)
print("Address difference between a.attr and b.attr : {}".format(id(a.attr)-id(b.attr)))
# Same addresses
b.attr = 0.
print("Address difference between after reassigning b.attr : {}".format(id(a.attr)-id(b.attr)))
# Different addresses

# Constructor
print()
print("=== Constructor ===")
print()
a = A()
b = A()
print("Address difference attributes : {}".format(id(a.attr)-id(b.attr)))
# Same addresses
print("Address difference instances : {}".format(id(a)-id(b)))
# Different addresses

class B:
	def __init__(self):
		self.attr = random.random()

a = B()
b = B()
print()
print("=== random init ===")
print()
print("Address difference attributes : {}".format(id(a.attr)-id(b.attr)))
# Different addresses
print("Address difference instances : {}".format(id(a)-id(b)))
# Different addresses