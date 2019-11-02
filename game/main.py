import os
import pygame
from pygame.locals import *
import math
from car import Car

TRACK_GREY = (108,108,108,255)
def rotate(surface, angle, pivot, offset):
    """Rotate the surface around the pivot point.

    Args:
        surface (pygame.Surface): The surface that is to be rotated.
        angle (float): Rotate by this angle.
        pivot (tuple, list, pygame.math.Vector2): The pivot point.
        offset (pygame.math.Vector2): This vector is added to the pivot.
    """
    rotated_image = pygame.transform.rotozoom(surface, -angle, 1)  # Rotate the image.
    rotated_offset = offset.rotate(angle)  # Rotate the offset vector.
    # Add the offset vector to the center/pivot point to shift the rect.
    rect = rotated_image.get_rect(center=pivot+rotated_offset)
    return rotated_image, rect  # Return the rotated image and shifted rect.

def collides(position, circuit, inColor=TRACK_GREY):
	return not circuit.get_at((round(position.x), round(position.y))) == inColor

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
	print(midPos)
	m = [int(midPos.x), int(midPos.y)]
	return m, dist(position, midPos)


pygame.init()
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0) 
BLUE = (0, 0, 128) 
FPS = 60
screen_size = (1024,768)

MAP_PATH = os.path.split(os.getcwd())[0] + "\\resources\\map.png"


screen = pygame.display.set_mode(screen_size)
circuit = pygame.image.load(MAP_PATH).convert_alpha()
clock = pygame.time.Clock()

font = pygame.font.SysFont("comicsansms", 40)


car = Car(0.,0,0)
running = True
while running:
	dtms = clock.tick(FPS)
	pygame.display.flip()
	running = car.inputs.update()
	renderedText = car.getStats()
	text = font.render(renderedText, True, (0, 128, 0))
	car.update(dtms)
	surface = car.render()
	surface = pygame.transform.rotate(surface, -math.degrees(car.heading))
	rot_rect = surface.get_rect()
	rot_rect.move(car.position.x, car.position.y)
	# surface_2, rect = rotate(surface, car.heading, pivot, offset)
	screen.fill(BLACK)
	screen.blit(circuit, (0,0))
	pygame.draw.rect(screen, GREEN, surface.get_rect())
	#screen.blit(surface, (car.position.x, car.position.y))
	screen.blit(surface, car.position)
	m, d = distanceToCollision(car.position, circuit, pygame.Vector2(1,0).rotate(math.degrees(car.heading)))
	pygame.draw.circle(screen, GREEN, m, 10)
	screen.blit(text, (0,0))
	pygame.display.flip()