import os
import pygame
import yaml
from car import Car

pygame.font.init()
PARAMS_FILE = "params.yaml"

def load_params(file):

	stream = open(file, 'r')
	dictionnary = list(yaml.load_all(stream))
	game_params = dictionnary[0]
	model_params = dictionnary[1]
	reward_params = dictionnary[2]

	vectors = [pygame.Vector2(tuple(pair)) for pair in game_params["vectors"]]
	game_params["vectors"] = vectors

	dummy_car = Car(0., pygame.Vector2(), load_image=False)
	car_dict = dummy_car.getCarDict()
	n_output = 0
	for output in game_params["car_output"]:
		o = car_dict[output]
		if type(o) is list:
			n_output += len(o)
		else:
			n_output += 1

	game_params["n_output"] = n_output
	neural_structure = [n_output + len(vectors)] + model_params["hidden_layers"] \
	+ [2 if model_params["linear_model"] else Inputs.size()]

	model_params["neural_structure"] = neural_structure
	
	return game_params, model_params, reward_params

game_params, model_params, reward = load_params(PARAMS_FILE)

TRACK_GREY = (108,108,108,255)
START_POINT = pygame.math.Vector2(455,237)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0, 255) 
BLUE = (0, 0, 255, 255)
RED = (255, 0, 0, 255)
FPS = game_params["fps"]
screen_size = tuple(game_params["screen_size"])

CP_REWARD = reward["cp_reward"] # reward when crossing a checkpoint
SCORE_DECAY = reward["score_decay"] # score decay per millisecond in circuit
DIE_PENALTY = reward["die_penalty"]
MAX_FRAMES_PENALTY = reward["max_frames_penalty"]
N_TOURS = reward["n_tours"]

sep = os.path.sep
MAP_PATH = os.path.split(os.getcwd())[0] + sep +"resources"+sep

font = pygame.font.SysFont("comicsansms", 40)

def load_circuit():
	circuit = pygame.image.load(MAP_PATH).convert_alpha()
	return circuit

