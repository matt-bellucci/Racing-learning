import pygame
import numpy as np
from constants import *
from car import Car
import copy
from circuit import Circuit
from game_loops import agent_inputs, game_loop
class Game:
	def __init__(self, vectors):
		pygame.init()
		self.screen = pygame.display.set_mode(screen_size)
		self.car = Car(0., START_POINT)
		self.circuit = Circuit()
		self.circuit_img = self.circuit.img_no_cp # pour tester les collisions, le checkpoint n'est pas important
		self.clock = pygame.time.Clock()
		self.running = True
		self.vectors = vectors
		self.score = 0.
		self.checkpoint = 0
		
	def get_agent_inputs(self):
		return agent_inputs(self.vectors, self.car, self.circuit_img)

	def run(self, inputs, render=True, is_ai=True):
		reward = 0.
		running = True
		if render:
			pygame.display.flip()
			self.screen.fill(BLACK)
			self.screen.blit(self.circuit_img, (0,0))
		self.car.inputs = inputs
		running, score_update, cp = game_loop(self.screen, self.clock, self.car, self.vectors,
			self.circuit, is_ai=is_ai, checkpoint=self.checkpoint, render_car=render)
		reward = score_update
		self.score += score_update
		self.checkpoint = cp
		if not running:
			self.die()
		if render:
			pygame.display.flip()

		return reward, running

	def die(self):
		self.car = Car(0., START_POINT)
		self.score = 0
		self.checkpoint = 0

