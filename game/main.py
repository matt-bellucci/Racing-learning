import os
import pygame
from pygame.locals import *
import math
from car import Car

pygame.init()
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0) 
BLUE = (0, 0, 128) 
FPS = 60
screen_size = (2000,100)



screen = pygame.display.set_mode(screen_size)
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
	rect_center = surface.get_rect().center
	print(rect_center)
	surface = pygame.transform.rotate(surface, -math.degrees(car.heading))
	screen.fill(BLACK)
	screen.blit(surface, (car.position.x, car.position.y))
	screen.blit(text, (0,0))
	pygame.display.update()