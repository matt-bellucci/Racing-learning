import copy
import random
class A:
	def __init__(self):
		self.attr = 0.

# Illustration copy.deepcopy
print("===Illustration copy.deepcopy===")
print()
a = A()
b = copy.deepcopy(a)
print("Difference adresse attribut sans changer b.attr : {}".format(id(a.attr)-id(b.attr)))
b.attr = 0.
print("Difference adresse attribut après changement b.attr : {}".format(id(a.attr)-id(b.attr)))


# Illustration constructeur
print("===Illustration constructeur===")
print()
a = A()
b = A()
print("Difference adresse attribut : {}".format(id(a.attr)-id(b.attr)))
print("Difference adresse instances : {}".format(id(a)-id(b)))

class B:
	def __init__(self):
		self.attr = random.random()

a = B()
b = B()
print("=Initialisation aléatoire=")
print()
print("Difference adresse attribut : {}".format(id(a.attr)-id(b.attr)))
print("Difference adresse instances : {}".format(id(a)-id(b)))