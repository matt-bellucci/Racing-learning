from datetime import datetime
from game import Game
from pygame import Vector2
from inputs import Inputs
import game_loops
import genetic
import pygame
import numpy as np
import utils
from constants import MAX_FRAMES_PENALTY
import matplotlib.pyplot as plt
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y-%H_%M")
vectors = [Vector2(0.,-1),Vector2(1,-1),
	Vector2(1,-0.5),Vector2(1,0.),
	Vector2(1,0.5),Vector2(1,1),Vector2(0.,1)]
INPUT_LEN = game_loops.INPUT_LEN_CAR + len(vectors)
neural_structure = [INPUT_LEN, 6, 2]
is_ai = True
n_frames = 0
epsilon = 1e-3
max_idle = 20
init_chromo = genetic.generate_chromos_from_struct(neural_structure)

def fit_function(indiv, model, max_frames=1000):
	def decision(state_vector):
		output_net = model.predict(x=np.array(state_vector).reshape(-1, len(state_vector)))
		decision_input = utils.net_to_input(output_net[0])
		return decision_input

	game = Game(vectors)
	score = 0
	n_idle = 0 
	t = 0
	while t < max_frames:
		pygame.event.get()
		network_inputs = game.get_agent_inputs()
		if np.abs(network_inputs).all()<epsilon:
			n_idle += 1
			if n_idle >= max_idle:
				score -= MAX_FRAMES_PENALTY
				break
		else:
			n_idle = 0
		inputs = decision(network_inputs)
		reward, running = game.run(inputs, is_ai=is_ai)
		t += 1
		if reward > 0:
			t = 0
		score += reward
		if not running:
			break

	if t>=max_frames-1:
		score -= MAX_FRAMES_PENALTY/(game.checkpoint+1)
	print(score)
	return score

n_indivs = 2
model = genetic.Genetic(n_indivs, neural_structure, fit_function)
mutation_start = 0.5
mutation_stop = 0.1
n_steps = 100
mutation_decay = (mutation_stop-mutation_start)/n_steps
for i in range(n_steps):
	mutation_chance = mutation_start + i*mutation_decay
	print("epsilon = ", mutation_chance)
	gen = model.train(mutation_chance=mutation_chance, n_bests=3)
	last_gen = model.generations[-2]
	print("best scores = ", last_gen.best_scores(n_firsts=5))
	filename = dt_string + str(i)
	# plt.hist([indiv.fitness for indiv in last_gen.individuals], 500)
	# plt.show()
	model.save_gen(filename+".hdf5")



