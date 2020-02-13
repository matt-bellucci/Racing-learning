# -*-coding:Latin-1 -*
import numpy as np
import copy
import os
import sys
from itertools import combinations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.layers import Dense, Activation
from keras.models import Sequential
import h5py


"""
Implémentation d'un algorithme génétique apprenant les poids d'un réseau de 
neurones densément connecté
"""

class Chromosome:
	"""
	Un chromosome est un ensemble de gènes, un chromosome représente 
	les caractéristiques d'un perceptron.
	Les gènes sont les poids entre les neurones de la couche précédente 
	et lui-même, avec un poids représentant le biais
	"""
	def __init__(self, size, gene_list=[], random_init=True, 
		start_range=-1, end_range=1):
		"""Constructeur d'un chromosome
		Un chromosome est une liste de gènes dont les valeurs sont comprises 
		dans un intervalle.
		Pour définir un réseau de neurones, 
		il faut une liste de liste de chromosomes. 

		On a donc [Couche1, Couche2, ...] où chaque couche est 
		une liste de chromosomes correspondant à chaque neurone de cette couche.
		Args:
			size (int): longueur de la liste
			gene_list (list): valeur des gènes si connus à l'avance, 
								si vide on en génère aléatoirement
			random_init (bool): si True, gènes choisis aléatoirement, 
								sinon initialisés à 0
			start_range (float): borne inférieur de l'intervalle
			end_range (float): borne supérieur de l'intervalle

		Returns:
			Chromosome : Liste de gènes
		"""
		# gestion des inputs 
		if start_range > end_range:
			self.start_range = end_range
			self.end_range = start_range
		else:
			self.start_range = start_range
			self.end_range = end_range

		if len(gene_list) == size:
			# clipping de la liste pour s'assurer d'etre bien dans 
			# l'intervalle donne
			self.genes = [max(min(end_range, gene_list[i]), start_range) 
				for i in range(size)]
			
		elif random_init:
			self.genes = start_range + np.random.rand(size)*(end_range-start_range)
		else:
			self.genes = np.zeros(size)

		self.genes = np.array(self.genes)


# ====== Définition de fonctions utiles pour gérer les chromosomes =====

def chromo_to_weights(chromos):
	""" Convertir chromosomes en poids pour réseau de neurones
	Fonction qui prend en entrée une liste de liste de chromosomes et 
	retourne une ensemble de poids que Keras peut utiliser 
	comme poids d'un réseau de neurones.

	Args:
		chromos (list(list(Chromosome))) : Liste de liste de chromosomes qui 
											représente le réseau de neurones

	Returns:

		numpy array: matrice des poids sous une forme lisible directement par Keras 

	"""
	weights_list = []
	for chromo in chromos:
		listed_chromos = np.array([x.genes for x in chromo], dtype=float)
		weights = [np.array(listed_chromos[:, :-1].T, dtype=float), 
			np.array(listed_chromos[:,-1], dtype=float)]
		weights_list = weights_list + weights
	return np.array(weights_list)

def weights_to_chromos(weights, start_range=-1., end_range=1.):
	""" Conversion poids d'un réseau en chromosomes
	Fonction inverse de chromo_to_weights, permet de générer les chromosomes 
	d'un Individu à partir des poids de son réseau

	Args:
		weights (list) : poids d'un réseau de neurones issus directement 
						du modèle Keras
		start_range (float) : borne inférieure des gènes de chaque chromosome
		end_range (float) : borne supérieure des gènes de chaque chromosome
	
	Returns:
		list(list(Chromosome)) : la liste des listes des chromosomes sous la 
									forme utilisée par Individu
	
	"""
	i = 0
	list_chromos = []
	while i<len(weights):
		# fonction inverse de chromo_to_weights donc opérations inverses
		current_array = weights[i].T
		current_array = np.c_[current_array, weights[i+1]]
		chromos = []
		for l in current_array:
			chromo = Chromosome(len(l), gene_list = l, start_range=start_range, 
				end_range=end_range, random_init=False)
			chromos.append(chromo)
		i += 2
		list_chromos.append(chromos)
	return list_chromos

	
def generate_chromos_from_struct(neural_structure):
	"""Genère les chromosomes d'un individu à partir de la structure du réseau
	Fonction utilisée pour initialiser les chromosomes des Individus 
	lors de la création du modèle génétique.
	
	Args:
		neural_structure (list): Le nombre de perceptrons par couche

	Returns:
		list(list(Chromosome)) : la liste des listes des chromosomes sous la 
									forme utilisée par Individu

	"""
	chromos = []
	for i in range(len(neural_structure)-1):
		chromos_layer = []
		for j in range(neural_structure[i+1]):
			chromos_layer.append(Chromosome(neural_structure[i]+1)) # nb connexions + biais
		chromos.append(chromos_layer)
	return chromos

class Individu:
	""" Un individu d'une espèce commune, ce qui le définit sont ses chromosomes
	et son score de fitness.

	La fit_function est la fonction de fitness qui doit etre codee de maniere
	a avoir un individu en entree et donc de creer la structure adaptee pour ensuite 
	calculer le score.

	On lui donne ses poids en attribut pour 
	"""
	def __init__(self, neural_structure, fit_function):
		# init avec gènes aléatoires
		self.chromos = generate_chromos_from_struct(neural_structure)
		# init avec random pour éviter des problèmes de partage d'adresse
		# mémoire
		self.fitness = np.random.random()
		self.fit_function = fit_function
		# simple conversion des chromosomes en poids pour le modèle
		self.weights = np.array(chromo_to_weights(self.chromos))
		self.neural_structure = neural_structure

	def update_fitness(self, model):
		"""
		Change les poids du modele general pour y mettre les poids
		de l'individu puis lance la fit_function pour determiner le score
		de l'individu
		"""
		model.set_weights(self.weights)
		self.fitness = self.fit_function(self, model)

	def copy(self):
		"""Copie d'un individu avec adresses mémoires différentes
		Fonction qui permet de copier un Individu en attribuant une
		adresse mémoire différente à l'instance mais aussi à chacun des
		attributs
		"""
		new = copy.deepcopy(self)
		new.chromos = copy.deepcopy(self.chromos)
		new.fitness = copy.deepcopy(self.fitness)
		new.weights = copy.deepcopy(self.weights)
		return new

	def save_model(self, filename):
		"""
		Sauvegarde des poids de l'individu
		"""
		file = h5py.File(filename, 'w')
		file.create_dataset(name="neural_structure", data=self.neural_structure)
		grp = file.create_group("weights")
		weights = np.array(self.weights)
		for k in range(len(weights)//2):
			l = 2*k
			grp.create_dataset(name="layer_{}".format(k), data=weights[l])
			grp.create_dataset(name="bias_{}".format(k), data=weights[l+1])
		file.flush()
		file.close()

class Population:
	"""Ensemble d'individus d'une même espèce
	La classe Population contient un tableau d'Individu
	et est capable de generer des Individu de la meme espece.
	"""
	def __init__(self, n_individus, neural_structure, fit_function, individuals=None):
		self.individuals = list()
		# si individuals est None, generation d'individus
		# sinon, si il y a trop ou pas asser d'individus dans individuals
		# on les genere aussi
		if not individuals or len(individuals) != n_individus:
			for i in range(n_individus):
				indiv = Individu(neural_structure, fit_function)
				self.individuals.append(indiv.copy())
		else:
			self.individuals = np.array(individuals)

		self.n_individus = n_individus
		self.structure = neural_structure # [n_inputs, n_neurons_layer1, n_neurons_layer2, ..., n_outputs]


	def fit(self, model):
	"""Calcul du score pour chaque individu de la population
	On utilise le meme modele neuronal pour tous les individus
	en changeant uniquement ce poids, ce qui est gere directement
	par l'individu
	"""
		for i in range(len(self.individuals)):
			print(str(i+1)+"/"+str(len(self.individuals)))
			self.individuals[i].update_fitness(model)

	def rank_fitness(self):
		"""
		Donne les indices des meilleurs individus
		tries dans l'ordre decroissant de score
		"""
		fitness = [indiv.fitness for indiv in self.individuals]
		sort = np.argsort(fitness)[::-1]
		return sort

	def best_scores(self, n_firsts=1):
		"""Affiche les scores des n meilleurs individus
		"""
		rank = np.array(self.rank_fitness())
		n_firsts = min(n_firsts, self.n_individus)
		sorted_indiv = np.array(self.individuals)[rank]
		return [indiv.fitness for indiv in sorted_indiv[:n_firsts]]

	def save_gen(self, filename):
		"""Sauvegarde des modeles de tous les individus de la population
		La sauvegarde se fait dans un fichier h5py avec les caracteristiques
		du reseau, le nombre d'individus, puis les poids de chaque individu dans un
		sous-groupe du fichier h5py
		"""
		file = h5py.File(filename, 'w')
		file.create_dataset(name="n_individus", data=self.n_individus)
		file.create_dataset(name="neural_structure", data=self.structure)
		for i in range(self.n_individus):
			grp = file.create_group("weights_{}".format(i))
			weights = np.array(self.individuals[i].weights)
			for k in range(len(weights)//2):
				l = 2*k
				grp.create_dataset(name="layer_{}".format(k), data=weights[l])
				grp.create_dataset(name="bias_{}".format(k), data=weights[l+1])
		file.flush()
		file.close()



def get_weights_per_couple(weights_per_parent, n_parents):
	"""Normalise les poids d'importance de chaque parents
	Les poids d'importance vont definir le nombre d'individus engendres par chaque parent.
	Plus un parent a un poids fort, plus il aura d'enfants. On calcule ensuite le poids
	d'importance par couple en multipliant simplement le poids du parent 1 par le poids du
	parent 2.
	On normalise ensuite tous ces poids pour avoir somme(poids) = 1
	
	Args :
		weights_per_parent (list(float)) : les poids attribues a chaque parent
					pas necessairement normalise
		n_parents (int) : le nombre de parents
	
	Returns:
		list(float) : liste des poids d'importance pour chaque couple, normalisee
	"""
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
			chromo.genes[:crossover_point] = indiv1.chromos[i][j].genes[:crossover_point]
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
				new_value = chromo.start_range + (chromo.end_range-chromo.start_range)*np.random.random()
				chromo.genes[gene_mutated] = new_value

	return new_indiv

def build_model(neural_structure):
	model = Sequential()
	model.add(Dense(neural_structure[1], input_dim=neural_structure[0], activation='softmax'))
	if len(neural_structure) > 2:
		for i, k in enumerate(neural_structure[2:-1]):
			model.add(Dense(k, activation='softmax'))
		model.add(Dense(neural_structure[-1], activation='linear'))
	model.compile(optimizer='sgd', loss='mean_squared_error')
	return model

class Genetic:

	def __init__(self, n_individus, neural_structure, fit_function):
		self.n_individus = n_individus
		self.generations = [Population(n_individus, neural_structure, fit_function)]
		self.neural_structure = neural_structure
		self.fit_function = fit_function
		self.n_generations = 0
		self.model = build_model(self.neural_structure)

	# Une optimisation est possible en n'iterant qu'une fois sur chaque chromosome de l'individu
	# Cependant, selon l'implementation, on n'itere pas forcement a chaque etape sur tout les chromosomes
	# Nous preferons restes generalistes en perdant en temps de calcul




	def generate_indivs(self, couple, mutation_chance=0.25):
		new_indiv = Individu(self.neural_structure, self.fit_function) # on prend le meme individu que le meilleur parent
		new_indiv = crossover(couple[0], couple[1], new_indiv)
		new_indiv = mutation(new_indiv, chance=mutation_chance)
		return new_indiv


	def train(self, n_bests=2, weights=None, mutation_chance=0.25, keep_n_bests=1):
		if n_bests < 2:
			n_bests = 2
		if keep_n_bests >= self.n_individus:
			keep_n_bests = n_individus - 1
		current_gen = self.generations[-1] # derniere generation
		print("Fit population")
		current_gen.fit(self.model)
		print("Fit ended")
		fitness_ranks = np.array(current_gen.rank_fitness())
		n_fittest = np.array(current_gen.individuals)[fitness_ranks]
		parents = [i for i in combinations(n_fittest[:n_bests], 2)] # toutes les paires possibles de parents, sans 2 fois le meme
		# la paire avec les meilleures parents est la premiere paire, puis qualite decroissante
		
		if not weights or len(weights) != n_bests:
			weights = np.ones(n_bests)	
		weights_per_couple = get_weights_per_couple(weights, n_bests)
		offspring_per_couple = get_offspring_per_couple(weights_per_couple, self.n_individus - keep_n_bests)
		print(offspring_per_couple)
		new_gen = n_fittest[:keep_n_bests].tolist()
		print(new_gen)
		print([ind.fitness for ind in new_gen])
		
		for i, couple in enumerate(parents):
			offsprings = []
			for j in range(offspring_per_couple[i]):
				new = Individu(self.neural_structure, self.fit_function)

				offsprings.append(self.generate_indivs(couple, mutation_chance=mutation_chance))
			new_gen += offsprings
		new_pop = Population(self.n_individus, self.neural_structure, self.fit_function, individuals=new_gen)
		
		self.generations.append(new_pop)
		self.n_generations += 1
		return self.generations[-1]


def load_gen(filename, fit_function, start_range=-1, end_range=1):
	file = h5py.File(filename, 'a')
	n_individus = file["n_individus"][()]
	neural_structure = file["neural_structure"][()]
	individuals = []
	for i in range(n_individus):
		grp = file["weights_{}".format(i)]
		n_layers = len(neural_structure)-1
		weights = []
		for k in range(n_layers):
			weights.append(np.array(grp["layer_{}".format(k)][()]))
			weights.append(np.array(grp["bias_{}".format(k)][()]))
		chromos = weights_to_chromos(weights, start_range=start_range, end_range=end_range)
		individual = Individu(neural_structure, fit_function)
		individual.chromos = chromos
		individual.weights = np.array(chromo_to_weights(chromos))
		individuals.append(individual)
	file.close()
	gen = Population(n_individus, neural_structure, fit_function, individuals=individuals)
	
	return gen

def load_model(filename, start_range=-1, end_range=1):
	file = h5py.File(filename, 'a')
	neural_structure = file["neural_structure"][()]	
	grp = file["weights"]
	n_layers = len(neural_structure)-1
	weights = []
	for k in range(n_layers):
		weights.append(np.array(grp["layer_{}".format(k)][()]))
		weights.append(np.array(grp["bias_{}".format(k)][()]))
	chromos = weights_to_chromos(weights, start_range=start_range, end_range=end_range)
	individual = Individu(neural_structure, lambda x: x)
	individual.chromos = chromos
	individual.weights = np.array(chromo_to_weights(chromos))
	file.close()
	return individual

def main():

	def fit(indi):
		return np.sum(indi.chromos[0][0].genes)

	neural_structure = [4,3]
	#print(neural_structure)
	init_chromo = generate_chromos_from_struct(neural_structure)
	print(init_chromo)
	weights = chromo_to_weights(init_chromo)
	#print(weights)
	conv_chromo = np.array(weights_to_chromos(weights))
	print(conv_chromo)
	conv_weights = np.array(chromo_to_weights(conv_chromo))
	#print(conv_weights)
	

if __name__== "__main__":
	main()
