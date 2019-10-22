from math import cos, sin, sqrt
from pygame.math import Vector2
import matplotlib.pyplot as plt
C_DRAG = 0.5
C_RR = 30 * C_DRAG

class Car:
	def __init__(self, x, y, mass=100):
		self.mass = mass
		self.position = Vector2(x, y)
		self.car_heading = 0.
		self.velocity = Vector2(0., 0.) # vecteur vitesse
		self.acceleration = Vector2(0., 0.)

	def speed(self):
		return sqrt(self.velocity.x**2 + self.velocity.y**2)

	def angle_vector(self):
		return Vector2(cos(self.car_heading), sin(self.car_heading))

	def traction(self, engine_force):
		return self.angle_vector() * engine_force

	def drag(self):
		return -C_DRAG * self.velocity * self.speed()

	def rolling_res(self):
		return -C_RR * self.velocity

	def longitudinal(self, engine_force):
		return self.traction(engine_force) + self.drag() + self.rolling_res()


	def update(self, dt, engine_force):
		self.acceleration = self.longitudinal(engine_force)/self.mass
		self.velocity += dt * self.acceleration
		self.position += dt * self.velocity

def main():
	foo = Car(0,0)

	for i in range(10):
		foo.update(0.5, 20)
		print(foo.position)
	for i in range(10):
		foo.update(0.5,0.)
		print(foo.position)
	for i in range(10):
		foo.update(0.5,-2.)
		print(foo.position)

if __name__ == "__main__":
	main()