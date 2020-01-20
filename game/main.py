import os
import pygame
from pygame.locals import *
import math
from constants import *
from car import Car
import utils
from circuit import Circuit
from game_loops import game_loop, agent_inputs
#TODO gestion score par voiture

pygame.init()
screen = pygame.display.set_mode(screen_size)
circuit = Circuit()
circuit_img = circuit.images[0] # pour tester les collisions, le checkpoint n'est pas important
circuit.display()
clock = pygame.time.Clock()
car = Car(0.,START_POINT)
running = True
vectors = [pygame.Vector2(1,0.)]
score = 0
checkpoint = 0
is_ai = False

while running:
	
	if is_ai:
		network_inputs = agent_inputs(vectors, car, circuit_img)
		car_inputs = agent_decision(network_inputs) # agent_decision retourne une classe Inputs,  
		# avec les entrees que l'IA a decide de faire. 
		car.inputs = car_inputs # actualisation des inputs de la voiture par l'ia
	running, score_update, checkpoint = game_loop(screen, clock, car, vectors, circuit, is_ai=is_ai,
	 checkpoint=checkpoint,
		render=True)
	score += score_update
	print("Score = ", score)
