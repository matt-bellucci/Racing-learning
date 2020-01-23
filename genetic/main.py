import cProfile
import numpy as np
from genetic import main

cProfile.run('main()')
#print(chromo_to_weights(init_chromo[0]))
#pop.fit()
#print("Génération modèle")
#gen = Genetic(100, neural_structure, init_chromo, fit)
#print("Entrainement")
#for i in range(10):
	#print("Génération " + str(i))
	#pop = gen.train(n_bests=4, weights=[0.5,0.2,0.2,0.1], n_proc = 8)
	#pop.fit()
	#print(pop.best_score())
	#print(pop.individuals[0].chromos[0][0].genes)