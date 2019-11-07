import os
from pygame.math import Vector2

TRACK_GREY = (108,108,108,255)
START_POINT = Vector2(455,237)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0) 
BLUE = (0, 0, 128) 
FPS = 60
screen_size = (1024,768)
MAP_PATH = os.path.split(os.getcwd())[0] + "\\resources\\map.png"
