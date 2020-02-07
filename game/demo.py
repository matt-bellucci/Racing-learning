from inputs import Inputs
import utils
import genetic
def demo(indiv=None, model, parameters):
	
	def decision(state_vector):
		output_net = model.predict(x=np.array(state_vector).reshape(-1, len(state_vector)))
		if len(output_net) == Inputs.size():
			inputs = utils.take_decision(output_net)
		elif len(output_net) == 2:
			inputs = utils.net_to_input(output_net)
		else:
			raise ValueError("Can manage only predictions of size {} or {}, list was of size {}".format(
				Inputs.size(), 2, len(output_net)))
		return inputs

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

	if t>=max_frames-1:
		score -= parameters['max_frames_penalty']/(game.checkpoint+1)
	print(score)
	return score


gen = genetic.load_gen(".\\models\\04-02-2020\\04_02_2020-15_346.hdf5", demo)
model_gen = genetic.build_model(gen.neural_structure)

