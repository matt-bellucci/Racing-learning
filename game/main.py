import os
import pygame
from pygame.locals import *
import math
from constants import *
from car import Car
import utils
from circuit import Circuit
from game_loops import game_loop
#TODO gestion score par voiture

pygame.init()
screen = pygame.display.set_mode(screen_size)
circuit = Circuit()
circuit.display()
clock = pygame.time.Clock()
car = Car(0.,START_POINT)
running = True
vectors = [pygame.Vector2(1,0.)]
score = 0
checkpoint = 0
while running:

	running, score_update, checkpoint = game_loop(screen, clock, car, vectors, circuit, checkpoint=checkpoint,
		render=True)
	score += score_update
	print("Score = ", score)
