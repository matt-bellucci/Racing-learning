from inputs import Inputs
import utils
import pygame
import genetic
import numpy as np
from game import Game
from constants import game_params, model_params
def decision_genetic(state_vector):
		output_net = model.predict(x=np.array(state_vector).reshape(-1, len(state_vector)))[0]
		if len(output_net) == Inputs.size():
			inputs = utils.take_decision(output_net)
		elif len(output_net) == 2:
			inputs = utils.net_to_input(output_net)
		else:
			raise ValueError("Can manage only predictions of size {} or {}, list was of size {}".format(
				Inputs.size(), 2, len(output_net)))
		return inputs

def demo(model, decision, parameters):
	game = Game(parameters['vectors'])
	score = 0
	n_idle = 0
	t = 0
	while t < parameters['max_frames']:
		pygame.event.get()
		network_inputs = game.get_agent_inputs()

		if np.abs(network_inputs).all()<parameters["epsilon"]:
			n_idle += 1
			if n_idle >= parameters["max_idle"]:
				score -= parameters['max_frames_penalty']
				break
		else:
			n_idle = 0
		inputs = decision(network_inputs)
		reward, running = game.run(inputs, is_ai=True)
		t += 1
		if reward > 0:
			t = 0
		score += reward
		if not running:
			break

	if t>=parameters["max_frames"]-1:
		score -= parameters['max_frames_penalty']/(game.checkpoint+1)
	print(score)
	return score

parameters = {}
parameters.update(game_params)
parameters.update(model_params)
print(parameters)
filename = ".\\models\\10_02_2020-15_08_s_208.hdf5"
# gen = genetic.load_gen(".\\models\\04-02-2020\\04_02_2020-15_346.hdf5", demo)
indiv = genetic.load_model(filename)
model = genetic.build_model(indiv.neural_structure)
model.set_weights(indiv.weights)

demo(model, decision_genetic, parameters)


