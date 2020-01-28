import numpy as np
import copy
import os
import sys
from itertools import combinations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.layers import Dense
from keras.models import Sequential


class Chromosome:

	def __init__(self, size, gene_list=[], random_init=True, start_range=0, end_range=1):
		self.start_range = start_range
		self.end_range = end_range
		if len(gene_list) == size:
			self.genes = [max(min(end_range, gene_list[i]), start_range) for i in range(size)]
			# clipping de la liste pour s'assurer d'etre bien dans l'intervalle donne
		elif random_init:
			self.genes = start_range + np.random.rand(size)*(end_range-start_range)
		else:
			self.genes = np.zeros(size)
		self.genes = np.array(self.genes)
		#print(self.genes)



def chromo_to_weights(chromos):
	# pour passer de chromosomes à poids d'un reseau, il faut donner a keras une liste [poids, biais] pour chaque couche
	listed_chromos = np.array([x.genes for x in chromos])
	weights = [listed_chromos[:, :-1].T, listed_chromos[:,-1]]
	return weights

def generate_chromos_from_struct(neural_structure):
	chromos = []
	for i in range(len(neural_structure)-1):
		chromos_layer = []
		for j in range(neural_structure[i+1]):
			chromos_layer.append(Chromosome(neural_structure[i]+1)) # nb connexions + biais
		chromos.append(chromos_layer)
	return chromos

class Individu:
	"""
	La fit_function est la fonction de fitness qui doit etre codee de maniere
	a avoir un individu en entree et donc de creer la structure adaptee pour ensuite 
	calculer le score.
	"""
	def __init__(self, chromos, neural_structure, fit_function):
		self.chromos = chromos
		self.fitness = 0.
		self.fit_function = fit_function
		self.structure = neural_structure
		self.model = self.__build_model__()

	def update_fitness(self):
		self.fitness = self.fit_function(self)

	def __build_model__(self):
		model = Sequential()
		first_weights = chromo_to_weights(self.chromos[0])

		model.add(Dense(self.structure[1], input_dim=self.structure[0], weights=first_weights, 
		activation='softmax'))
		if len(self.structure) > 2:
			for i, k in enumerate(self.structure[2:]):
		 		model.add(Dense(k, weights=chromo_to_weights(self.chromos[i+1]), activation='softmax'))
		model.compile(optimizer='sgd', loss='mean_squared_error')
		return model





class Population:

	def __init__(self, n_individus, neural_structure, initial_chromosomes, fit_function, individuals=None):
		if not individuals or len(individuals) != n_individus:
			self.individuals = np.array([Individu(copy.deepcopy(initial_chromosomes), neural_structure, fit_function) for x in range(n_individus)])
		else:
			self.individuals = np.array(individuals)
		# deepcopy des chromosomes pour qu'ils soient modifiables, sinon c'est une reference a la meme adresse
		# donc les modifier pour 1 indiv les modifient pour tous
		self.n_individus = n_individus
		self.structure = neural_structure # [n_inputs, n_neurons_layer1, n_neurons_layer2, ..., n_outputs]


	def fit(self):
		for individual in self.individuals:
			individual.update_fitness()

	def rank_fitness(self):
		return sorted([i for i in range (self.n_individus)], reverse=True, key = lambda x : self.individuals[x].fitness)

	def best_scores(self, n_firsts=1):
		rank = np.array(self.rank_fitness())
		return [self.individuals[rank[i]].fitness for i in range(n_firsts)]



def get_weights_per_couple(weights_per_parent, n_parents):
	idx_combi = [i for i in combinations(np.arange(n_parents), 2)]
	weights_per_couple = [weights_per_parent[p[0]]*weights_per_parent[p[1]] for p in idx_combi]
	weights_per_couple = np.array(weights_per_couple)/np.sum(weights_per_couple)
	return weights_per_couple

def get_offspring_per_couple(weights_per_couple, n_individus):
	offspring_per_couple = []
	count = 0
	for w in weights_per_couple:
		n_offspring = int(np.ceil(w*n_individus))
		if count + n_offspring >= n_individus:
			n_offspring = n_individus - count
			offspring_per_couple.append(n_offspring)
			count += n_offspring
			pass
		else:
			count += n_offspring
			offspring_per_couple.append(n_offspring)
	return offspring_per_couple


def crossover(indiv1, indiv2, new_indiv):
	"""
	indiv1, indiv2 : Individus parents
	new_indiv : Enfant
	On prend une partie des genes du premier parent jusqu'a un point aleatoire, puis on prend
	les genes du 2e parent apres ce point.
	"""
	for i, layer_chromos in enumerate(new_indiv.chromos):
		for j, chromo in enumerate(layer_chromos):
			crossover_point = np.random.randint(len(chromo.genes)) # point ou l'on prend les genes de l'autre individu
			chromo.genes[crossover_point:] = indiv2.chromos[i][j].genes[crossover_point:]

	return new_indiv

def mutation(new_indiv, chance=0.25):

	# plusieurs choix d'implementation, mutation chromosome par chromosome ou sur un gene parmi tous
	# ici on fera chromosome par chromosome
	for layer_chromos in new_indiv.chromos:
		for chromo in layer_chromos:
			eps = np.random.rand()
			if eps < chance:
				gene_mutated = np.random.randint(len(chromo.genes))
				new_value = chromo.start_range + np.random.rand() * (chromo.end_range - chromo.start_range)
				chromo.genes[gene_mutated] = new_value

	return new_indiv

class Genetic:

	def __init__(self, n_individus, neural_structure, initial_chromosomes, fit_function):
		self.n_individus = n_individus
		self.generations = [Population(n_individus, neural_structure, initial_chromosomes, fit_function)]
		self.neural_structure = neural_structure
		self.fit_function = fit_function
		self.n_generations = 0

	# Une optimisation est possible en n'iterant qu'une fois sur chaque chromosome de l'individu
	# Cependant, selon l'implementation, on n'itere pas forcement a chaque etape sur tout les chromosomes
	# Nous preferons restes generalistes en perdant en temps de calcul




	def train(self, n_bests=2, weights=None, mutation_chance=0.25):
		if n_bests < 2:
			n_bests = 2
		current_gen = self.generations[-1] # derniere generation
		current_gen.fit()
		fitness_ranks = np.array(current_gen.rank_fitness())
		n_fittest = np.array(current_gen.individuals)[fitness_ranks[:n_bests]]
		parents = [i for i in combinations(n_fittest[:n_bests], 2)] # toutes les paires possibles de parents, sans 2 fois le meme
		# la paire avec les meilleures parents est la premiere paire, puis qualite decroissante
		
		if not weights or len(weights) != n_bests:
			weights = np.ones(n_bests)	
		weights_per_couple = get_weights_per_couple(weights, n_bests)
		offspring_per_couple = get_offspring_per_couple(weights_per_couple, self.n_individus)
		new_gen = []

		n_ite = len(parents)*self.n_individus
		ite = 0
		for i, couple in enumerate(parents):
			offsprings = []
			for j in range(offspring_per_couple[i]):
				sys.stdout.write("\r {:.0f}%".format(ite*100/n_ite))
				sys.stdout.flush()
				offsprings.append(generate_indivs(couple, mutation_chance=mutation_chance))
				ite += 1
			new_gen += offsprings
		new_pop = Population(self.n_individus, self.neural_structure, None, self.fit_function, individuals=new_gen)
		
		self.generations.append(new_pop)
		self.n_generations += 1
		return self.generations[-1]

def generate_indivs(couple, mutation_chance=0.25):
	new_indiv = copy.deepcopy(couple[0]) # on prend le meme individu que le meilleur parent
	new_indiv = crossover(couple[0], couple[1], new_indiv)
	new_indiv = mutation(new_indiv, chance=mutation_chance)
	return new_indiv

def main():

	def fit(indi):
		return np.sum(indi.chromos[0][0].genes)

	neural_structure = [4,10,5,3]
	print(neural_structure)
	init_chromo = generate_chromos_from_struct(neural_structure)
	#print(type(init_chromo))
	
	pop = Population(100, neural_structure, init_chromo, fit)
	#print(chromo_to_weights(init_chromo[0]))
	#pop.fit()
	print("Génération modèle")
	#gen = Genetic(100, neural_structure, init_chromo, fit)
	print("Entrainement")
	# for i in range(10):
	# 	print("Génération " + str(i))
	# 	pop = gen.train(n_bests=4, weights=[0.5,0.2,0.2,0.1], n_proc = 8)
	# 	pop.fit()
	# 	print(pop.best_score())
	# 	print(pop.individuals[0].chromos[0][0].genes)

	#ranks = pop.rank_fitness()
	#print([pop.individuals[i].fitness for i in ranks])
	#model = Genetic(10,init_chromo, fit)
	#model.train()

if __name__== "__main__":
	main()
