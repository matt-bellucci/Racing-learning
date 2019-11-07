import car
from pygame.math import Vector2
class Outputs:


	def __init__(self, car_data, sensor_vectors):
		"""
		car_data : liste de string des donnees a recuperer de la voiture (comme la vitesse etc)
		sensor_vectors : liste des directions des capteurs pour detecter la distance a la prochaine collision
		"""

		self.car_data = car_data
		self.sensor_vectors = sensor_vectors
		self.collision_distances = []
		self.car_outputs = []
		self.outputs = []

	def get_collision_distances(self, car):
		self.collision_distances = []
		for vect in sensor_vectors:
			


