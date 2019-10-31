import os
import pygame
from pygame.locals import *
import math
from car import Car

screen_size = (1024,768)
screen = pygame.display.set_mode(screen_size)
screen.fill((0,192,0))
car = Car(0.,0,0)
clock = pygame.time.Clock()
running = True
while running:
	clock.tick(24)
	frames = 0
	frames += 1
	pygame.display.flip()
	running = car.inputs.update()
	car.updatePhysics(1)