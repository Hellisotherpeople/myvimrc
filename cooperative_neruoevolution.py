import numpy as np
import math
import numpy.ma as ma
from random import randint, random
import random
import pdb
# from numba import jit, prange, njit
from scipy.special import expit
from sklearn.datasets import make_classification, make_regression, make_moons, make_blobs
import scipy.linalg.blas as FD
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.switch_backend('agg')

X, y = make_moons(n_samples=1000, random_state=42)
# X, y = make_regression(n_samples=100, n_features=2, n_informative=1, random_state=43)
org_y = y
y = np.asarray([[i] for i in np.absolute(y)])
# X = np.absolute(X) * 8  # make it a reasonable
np.set_printoptions(suppress=True)


class Neural_Network(object):
    def __init__(self, num_population, number_iteration, mutation_probability):
        # parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 30
        self.w1_population_list = num_population * [None]
        self.w2_population_list = num_population * [None]
        self.num_population = num_population
        self.w1_size = self.inputSize * self.hiddenSize
        self.w2_size = self.hiddenSize * self.outputSize
        self.total_size = self.w1_size + self.w2_size
        self.W1 = []
        self.W2 = []
        self.pop_matrix = self.initialize_population()
        self.mutation_probability = mutation_probability
        self.iterator = 0

    def forward(self, X):
        # forward propagation through our network
        # dot product of X (input) and first set of 3x2 weights
        #self.z = np.dot(X, self.W1)
        self.z = FD.dgemm(alpha=1.0, a=X.T, b=self.W1.T,
                          trans_a=True, trans_b=True)  # MKL fast dot product
        self.z2 = NN.Elu((self.z))  # ACTIVATION FUNCTION
        # dot product of hidden layer (z2) and second set of 3x1 weights
        #self.z3 = np.dot(self.z2, self.W2)
        self.z3 = FD.dgemm(alpha=1.0, a=self.z2.T,
                           b=self.W2.T, trans_a=True, trans_b=True)
        o = expit((self.z3))  # final activation function
        return o

    def crossover(self, parent1, parent2):
        # parent 1 and parent 2 shape should be the same, but checking for this is
        # expensive...
        rand_list = np.random.randint(0, 2, size=parent1.shape)
        rand_list_2 = np.random.randint(0, 2, size=parent1.shape)
        child1 = np.multiply(rand_list, parent1) + \
            np.multiply(1-rand_list, parent2)
        child2 = np.multiply(rand_list_2, parent1) + \
            np.multiply(1-rand_list_2, parent2)
        return(child1, child2)

    def mutate(self, arr, probability):
        temp = np.asarray(arr)   # Cast to numpy array
        shape = temp.shape       # Store original shape
        temp = temp.flatten()    # Flatten to 1D
        num_to_change = int(len(temp) * probability)
        inds = np.random.choice(
            temp.size, size=num_to_change)   # Get random indices
        # multiply weights by random # from -2 to 2)
        temp[inds] = temp[inds] + np.random.uniform(-2, 2, size=num_to_change)
        temp = temp.reshape(shape)                     # Restore original shape
        return temp

    def initialize_population(self):
        pop_arr = np.random.rand(self.num_population, self.total_size, 2)
        for elem in pop_arr:
            for neuron in elem:
                neuron[1] = float(1)
        return pop_arr

    def convert_to_weights(self, i, update):
        weights = self.pop_matrix[i, :, 0]
        fitnes_values = self.pop_matrix[i, :, 1]
        #w1_fitness_slice = fitnes_values[:self.w1_size]
        #w2_fitness_slice = fitnes_values[self.w1_size:self.total_size]
        w1_slice = weights[:self.w1_size]
        w2_slice = weights[self.w1_size:self.total_size]
        self.W1 = np.reshape(w1_slice, (self.inputSize, self.hiddenSize))
        self.W2 = np.reshape(w2_slice, (self.hiddenSize, self.outputSize))
#        print(self.W1)
#        print(self.W2)
        if update:
            prediction = self.forward(X)
            error = self.rmse(prediction, y)
            np.place(fitnes_values, fitnes_values > error,
                     error)  # update the fitness values

        #pop_matrix = list(np.ravel(pop_matrix))
        #the_list = self.total_size * [None]
        # for i in range(0, self.total_size):
        #    the_list[i] = [pop_matrix[i], 0]
        # print(the_list[i][0])


#        for i in range(0, self.num_population):
#            self.w1_population_list[i] = pop_matrix[i][0:self.w1_size].reshape(self.inputSize, self.hiddenSize)

#        for i in range(0, self.num_population):
#            self.w2_population_list[i] = pop_matrix[i][self.w1_size:self.total_size].reshape(self.hiddenSize, self.outputSize)

        # print(self.w1_population_list)

    def plot_decision_boundary(self):
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.3
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.c_[xx.ravel(), yy.ravel()] ##I'm not good enough at numpy to say that I just "knew" that this is what I wanted.. thank you stack overflow
        Z = NN.forward(Z)
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        cp = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r)
        if self.iterator == 1:
            plt.colorbar(cp)
        plt.scatter(X[:, 0], X[:, 1], c=org_y, cmap=plt.cm.Spectral)

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def ReLU(self, arr):
        arr[arr < 0] = 0
        return arr

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def Elu(self, x, a=2):
        """exponential linear unit, from this paper  https://arxiv.org/abs/1511.07289... seems to work quite well"""
        return np.where(x <= 0, a * (np.exp(x) - 1), x)

    def gaussian(self, x):
        sq = np.square(x)
        neg = np.negative(sq)
        return np.exp(neg)

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def cooperative_neuroevolution(self):
        fitness_vals = self.pop_matrix[:, :, 1]
        weight_vals = self.pop_matrix[:, :, 0]
        # print(a)
        # the average fitness of each network
        best_fitness = 1
        while self.iterator < 50000000000:
            network_avg_fitness = fitness_vals.sum(axis=1) / self.total_size
            # print(network_avg_fitness)
            most_fit_networks_i = network_avg_fitness.argpartition(
                    2)[:2]  # get smallest N fitnesses indicies
            worst_network_i = network_avg_fitness.argpartition(-1)[-1:]
            # the actual networks
            most_fit_networks = fitness_vals[most_fit_networks_i]
            most_fit_networks_weight = weight_vals[most_fit_networks_i]
            least_fit_networks = fitness_vals[worst_network_i]
            least_fit_networks_weight = weight_vals[worst_network_i]

            for i in range(1, len(most_fit_networks_i)):
                self.convert_to_weights(most_fit_networks_i[i-1], False)
                w1_first_parent = self.W1
                w2_first_parent = self.W2
                best_pred = self.forward(X)
                check_fitness = self.rmse(best_pred, y)
                if best_fitness > check_fitness:
                    best_fitness = check_fitness
                    self.plot_decision_boundary()
                    plt.pause(0.001)
                    print("best fitness " + str(best_fitness) + " generation: " + str(self.iterator))
                self.convert_to_weights(most_fit_networks_i[i], False)
                w1_second_parent = self.W1
                w2_second_parent = self.W2
                w1_child1, w1_child2 = self.crossover(
                    w1_first_parent, w1_second_parent)
                w2_child1, w2_child2 = self.crossover(
                    w2_first_parent, w2_second_parent)
                if random.random() < self.mutation_probability:
                    w1_child1 = self.mutate(w1_child1, 0.8)
    #                w1_child2 = self.mutate(w1_child2, 0.3)
                    w2_child1 = self.mutate(w2_child1, 0.8)
    #                w2_child2 = self.mutate(w2_child2, 0.3)
            self.W1 = w1_child1
            self.W2 = w2_child1
            new_prediction_1 = self.forward(X)
            new_fitness_1 = self.rmse(new_prediction_1, y)
            new_weights_1 = np.concatenate((w1_child1.ravel(), w2_child1.ravel()))
            #self.W1 = w1_child2
            #self.W2 = w2_child2
            #new_prediction_2 = self.forward(X)
            #new_fitness_2 = self.rmse(new_prediction_2, y)
            #new_weights_2 = np.concatenate((w1_child2.ravel(), w2_child2.ravel()))# end of recombine 
            # print(new_weights)
            #print(fitness_vals)
            weight_vals[worst_network_i] = new_weights_1 #new weight replacement!
            #np.apply_along_axis(fitness_vals[worst_network_i], 0
            for k in range(0, self.total_size):
                fitness_vals[worst_network_i, k] = np.mean([fitness_vals[worst_network_i, k],new_fitness_1]) #fitness replacement
            #print(fitness_vals)
            #print(worst_network_i)
            #print(self.pop_matrix[:,0:1,0])
            for j in range(0, self.total_size):
                np.random.shuffle(self.pop_matrix[:,j,:])#permutate all values
            self.iterator+=1
            #print(self.pop_matrix[:,0:1,0])

'''
        # lowest fitness is furthest to the right
        num_to_replace = -math.floor(self.total_size/3) #TODO: REPLACE THE BAD WEIGHTS IN THE SUBPOPULATIONS, NOT IN EACH NETWORK:w
        least_fit_sorted_networks = []
        for fit_network in most_fit_networks:
            least_fit_sorted_networks.append(fit_network.argsort(axis=0))
        least_fit_sorted_networks = np.asarray(least_fit_sorted_networks)
        to_replace = least_fit_sorted_networks[:, num_to_replace:]
        flipped_replace = np.flip(to_replace, axis=1)
        print(flipped_replace)
        print("------------------")
        k = 1  # because I don't know how to index better
        # print the worst performing neurons
        print(most_fit_networks[k:k+1, flipped_replace[k]])
        print(new_weights_1)
        # the weights we will replace it with
        len_weight = len(new_weights_1[flipped_replace[0]])
        print(len_weight)
        # np.copyto(most_fit_networks[k:k+1, flipped_replace[k]], new_weights_1[flipped_replace[0]]) #replace elements
        # replace weights, but we still have to update values
        broad = np.repeat(new_fitness_1, len_weight, axis=0)
        most_fit_networks_weight[:, flipped_replace[k]
                                 ] = new_weights_1[flipped_replace[0]] #replace weights 
        most_fit_networks[:, flipped_replace[k]] = np.mean((most_fit_networks[:, flipped_replace[k]], broad), axis=0) #and replace values (possibly buggy) 
        weight_vals[most_fit_networks_i]= most_fit_networks_weight
        fitness_vals[most_fit_networks_i] = most_fit_networks
        #print(fitness_vals[most_fit_networks_i])
        #print(fitness_vals)
        #shuffled = np.random.shuffle(self.pop_matrix)
        print(self.pop_matrix[:,0,:])
'''
NN = Neural_Network(10, 100, 5)

for i in range(0, 10):
    NN.convert_to_weights(i, True)
NN.cooperative_neuroevolution()


# recombine()


# print(a.sort(axis=1)) #sort the weights in each network with no array copy!


# print(NN.w1_population_list[1])
#print(NN.forward(X, NN.w1_population_list[1], NN.w2_population_list[2]))
