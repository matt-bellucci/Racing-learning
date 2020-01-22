from game import Game
from pygame import Vector2
from inputs import Inputs
vectors = [Vector2(0.,-1),Vector2(1,-1),
	Vector2(1,-0.5),Vector2(1,0.),
	Vector2(1,0.5),Vector2(1,1),Vector2(0.,1)]
n_cars = 2
game = Game(n_cars, vectors)
is_ai = False
n_frames = 100000
main_inputs = Inputs()
inputs = [Inputs() for i in range(n_cars)]
print(len(inputs))
for t in range(n_frames):
	network_inputs = game.get_agent_inputs()
	stops = []
	stops.append(main_inputs.update())
	for i in range(n_cars):
		stops.append(inputs[i].update())
	if not all(stops):
		print("Stop")
		break
	rewards, runnings = game.run(inputs, is_ai=is_ai)


