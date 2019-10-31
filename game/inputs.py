import os
import pygame
from pygame.locals import *
import math

class Inputs:
	def __init__(self):
		pygame.key.set_repeat(1) # if any key is kept down, pygame will register it every 1ms
		self.left = 0.
		self.right = 0.
		self.throttle = 0.
		self.brake = 0.
		self.ebrake = 0.

	def reset(self):
		self.left = 0.
		self.right = 0.
		self.throttle = 0.
		self.brake = 0.
		self.ebrake = 0.

	def update(self):
		self.reset()
		key = pygame.key.get_pressed()

		for event in pygame.event.get():

			if event.type == pygame.KEYDOWN:

				if event.key == pygame.K_UP:
					self.throttle += 1.

				if event.key == pygame.K_DOWN:
					self.brake = 1.

				if event.key == pygame.K_LEFT:
					self.left = 1.

				if event.key == pygame.K_RIGHT:
					self.right = 1.

				if event.key == pygame.K_SPACE:
					self.ebrake = 1.

				self.display()

			if event.type == pygame.QUIT:
				print("Stop")
				return False
		return True

	def display(self):
		print("Left = {}\nRight = {}\nThrottle = {}\nBrake = {}\nEBrake = {}"
			.format(self.left, self.right, self.throttle, self.brake, self.ebrake))



