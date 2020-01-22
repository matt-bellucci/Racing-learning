import pygame
import numpy as np
from constants import *
from car import Car
import copy
from circuit import Circuit
from game_loops import agent_inputs, game_loop
class Game:
	def __init__(self, n_cars, vectors):
		pygame.init()
		self.screen = pygame.display.set_mode(screen_size)
		self.n_cars = n_cars
		self.cars = [Car(0., START_POINT) for i in range(n_cars)]
		self.circuit = Circuit()
		self.circuit_img = self.circuit.img_no_cp # pour tester les collisions, le checkpoint n'est pas important
		self.clock = pygame.time.Clock()
		self.running = True
		self.vectors = vectors
		self.scores = np.zeros(n_cars)
		self.checkpoints = np.zeros(n_cars, dtype=int)
		
	def get_agent_inputs(self):
		return [agent_inputs(self.vectors, car, self.circuit_img) for car in self.cars]

	def run(self, inputs, render=True, is_ai=True):
		if is_ai and (len(inputs) != self.n_cars):
			raise ValueError("Number of inputs does not match number of cars")
		rewards = np.zeros(self.n_cars)
		runnings = [True for i in range(self.n_cars)]
		if render:
			pygame.display.flip()
			self.screen.fill(BLACK)
			self.screen.blit(self.circuit_img, (0,0))
		for i in range(self.n_cars):
			car = self.cars[i]
			car.inputs = inputs[i]
			cp = self.checkpoints[i]
			running, score_update, checkpoint = game_loop(self.screen, self.clock, car, self.vectors,
				self.circuit, is_ai=is_ai, checkpoint=cp, render_car=render)
			rewards[i] = score_update
			self.scores[i] += score_update
			self.checkpoints[i] = checkpoint
			runnings[i] = running
			if not running:
				self.die(i)
		if render:
			pygame.display.flip()

		return rewards, runnings

	def die(self, i):
		self.cars[i] = Car(0., START_POINT)
		self.scores[i] = 0
		self.checkpoints[i] = 0

