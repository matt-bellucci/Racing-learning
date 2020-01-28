from game import Game
from pygame import Vector2
from inputs import Inputs
import game_loops
import genetic
import pygame
vectors = [Vector2(0.,-1),Vector2(1,-1),
	Vector2(1,-0.5),Vector2(1,0.),
	Vector2(1,0.5),Vector2(1,1),Vector2(0.,1)]
INPUT_LEN = game_loops.INPUT_LEN_CAR + len(vectors)
neural_structure = [INPUT_LEN, 6, Inputs.size()]
is_ai = False
n_frames = 0

init_chromo = genetic.generate_chromos_from_struct(neural_structure)

def fit_function(indiv, max_frames=10000):
	def decision(state_vector):
		output_net = indiv.model.predict(x=np.array(state_vector).reshape(-1, len(state_vector)))
		decision_input = utils.take_decision(output_net)
		return decision_input
	game = Game(1, vectors)
	score = 0
	for t in range(n_frames):
		network_inputs = game.get_agent_inputs()
		inputs = decision(network_inputs)
		rewards, runnings = game.run(inputs, is_ai=is_ai)
		score += rewards
		if not runnings[0]:
			break
	return score
n_indivs = 30
model = genetic.Genetic(n_indivs, neural_structure, init_chromo, fit_function)
mutation_start = 0.9
mutation_stop = 0.2
n_steps = 10
mutation_decay = (mutation_stop-mutation_start)/n_steps
for i in range(n_steps):
	mutation_chance = mutation_start + i*mutation_decay
	gen = model.train(mutation_chance=mutation_chance)
	print(gen.best_score(n_firsts=5))

