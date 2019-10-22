import os
import pygame
from pygame.locals import *
import math
#import car

screen_size = (1024,768)
screen = pygame.display.set_mode(screen_size)
screen.fill((0,192,0))

clock = pygame.time.Clock()
running = True

while running:
	clock.tick(24)
	frames = 0
	frames += 1

	pygame.display.flip()

	key = pygame.key.get_pressed()
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
			print("Escape")

