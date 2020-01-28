import math
import pygame
import car
from constants import *
from inputs import Inputs
def distanceToCollision(position, circuit, direction, checkDistance=1, precision=10, max_iter=10):
	# print(direction)
	startPos = pygame.Vector2(position.x, position.y)
	w,h = circuit.get_size()
	endPos = startPos
	while not collides(endPos, circuit):
		endPos += direction * checkDistance
		if endPos.x > w:
			endPos.x = w-1
		if endPos.y > h:
			endPos.y = h-1
		if endPos.x < 0:
			endPos.x = 0
		if endPos.y < 0:
			endPos.y = 0
	dist = lambda p1, p2: math.hypot(p1.x - p2.x, p1.y - p2.y)

	i = 0
	while (i < max_iter) and (dist(startPos, endPos) > precision):
		midPos = (endPos+startPos)/2

		if not collides(midPos, circuit):
			startPos = midPos
		else:
			endPos = midPos
		i += 1
	midPos = (endPos+startPos)/2
	# print(midPos)
	m = [int(midPos.x), int(midPos.y)]
	return m, dist(position, midPos)

def collides(position, circuit, inCircuitColors=[TRACK_GREY, RED, GREEN]):
	return not circuit.get_at((int(round(position.x)), int(round(position.y)))) in inCircuitColors

def onCheckpoint(position, circuit, checkpointColor=GREEN):
	# print(circuit.get_at((int(round(position.x)), int(round(position.y)))))
	return circuit.get_at((int(round(position.x)), int(round(position.y)))) == checkpointColor

def take_decision(decision):
	"""
	Liste de sortie du reseau de chaque individu
	Si avancer > freiner alors on ne garde que avancer normalisé a 1
	De meme pour gauche/droite

	Cela permet a l'algo genetique de ne pas avoir a donner de valeurs exactes et facilite l'apprentissage
	"""
	decision_input = Inputs.list_to_inputs(decision)
	if decision_input.left > decision_input.right:
		decision_input.left = 1
		decision_input.right = 0
	else:
		decision_input.left = 0
		decision_input.right = 1
	if decision_input.throttle > decision_input.brake + decision_input.ebrake:
		decision_input.throttle = 1
		decision_input.brake = 0
		decision_input.brake = 0
	else:
		decision_input.throttle = 0
		decision_input.brake = 1
		# on garde la valeur initiale de frein a main
	return decision_input
	

