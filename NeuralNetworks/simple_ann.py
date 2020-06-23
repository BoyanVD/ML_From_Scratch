import numpy as np 
from neural_network_exception import NeuralNetworkInputException

class SimpleNeuralNetwork :
	""" Simple three neuron layers and two weight layers Neural Network, implemented from scratch using pure python 3 and numpy.

	Attributes:
		layers_sizes -- List that represents the sizes of every layer in the NN, counting input and output layers as well.
		weights -- list of numpy arrays, containing the weights for every layer in the NN
		biases -- list of numpy arrays, containing the bias terms of every layer in the NN
	"""


	def __init__(self, layers_sizes) :
		if len(layers_sizes) != 3:
			raise NeuralNetworkInputException("Layers sizes list must be with length 3 !")


		self.weights = []
		self.biases = []
		self.layers_sizes = layers_sizes
		self.initializeWeightsAndBiases()

	def initializeWeightsAndBiases(self):
		for layer_index in range(len(self.layers_sizes) - 1) :
			weight = np.random.randn(self.layers_sizes[layer_index], self.layers_sizes[layer_index + 1])
			bias = np.random.randn(self.layers_sizes[layer_index + 1])

			self.weights.append(weight)
			self.biases.append(bias)

	@staticmethod
	def sigmoid(a):
		return 1 / (1 + np.exp(-a))

	@staticmethod
	def softmax(a):
		expA = np.exp(a)
		return expA / expA.sum(axis = 1, keepdims = True)

	def feedforward(self, X):
		hidden_layer_activation_input = X.dot(self.weights[0]) + self.biases[0]

		Z = SimpleNeuralNetwork.sigmoid(hidden_layer_activation_input)
		output_layer_activation_input = Z.dot(self.weights[1] + self.biases[1])

		Y = SimpleNeuralNetwork.softmax(output_layer_activation_input)

		return Y, Z

	def derivative_wrt_w2(self, Z, T, Y):
		return Z.T.dot(T - Y)

	def derivative_wrt_b2(self, T, Y):
		return (T - Y).sum(axis = 0)

	def derivative_wrt_w1(self, Z, T, Y, X):
		return X.T.dot((T - Y).dot(self.weights[1].T) * Z * (1 - Z))

	def derivative_wrt_b1(self, Z, T, Y):
		return ((T - Y).dot(self.weights[1].T) * Z * (1 - Z)).sum(axis = 0)

	def cost_function(self, T, Y):
		total = T * np.log(Y)
		return total.sum()

	def classification_rate(Y, P):
		correct_pred = 0
		total_pred = len(Y)

		for i in range(total_pred):
			if Y[i] == P[i]:
				correct_pred += 1

		return float(correct_pred) / total_pred

	def train(self, X, Y, epochs = 1000, learning_rate = 0.01):
		D = self.layers_sizes[0] # Input dimensionality
		M = self.layers_sizes[1] # Hidden layer number of neurons
		K = self.layers_sizes[2] # Number of output classes
		N = X.shape[0] # Number of samples
		costs = [] # List of costs through train process

		# Constructing the indicator matrix, using one-hot encoding.
		T = np.zeros((N, K))
		for i in range(N):
			T[i, Y[i]] = 1 

		for epoch in range(epochs):
			output, hidden = self.feedforward(X)

			if epoch % 100 == 0:
				cost = self.cost_function(T, output)
				costs.append(cost)

			self.weights[1] += learning_rate * self.derivative_wrt_w2(hidden, T, output)
			self.biases[1] += learning_rate * self.derivative_wrt_b2(T, output)

			self.weights[0] += learning_rate * self.derivative_wrt_w1(hidden, T, output, X)
			self.biases[0] += learning_rate * self.derivative_wrt_b1(hidden, T, output)

		return costs

	def predict(self, X):
		output, hidden = self.feedforward(X)
		P_Y = np.argmax(output, axis = 1)

		return P_Y

import matplotlib.pyplot as plt 

def main():
	N_per_class = 500
	D = 2 # input dimensionality
	M = 3 # number of neurons in hidden layer
	K = 3 # number of classes

	# Generating three Gaussian clouds
	X1 = np.random.randn(N_per_class, D) + np.array([0, -2]) # Fist cloud
	X2 = np.random.randn(N_per_class, D) + np.array([2, 2]) # Second cloud
	X3 = np.random.randn(N_per_class, D) + np.array([-2, 2]) # Third cloud

	X = np.vstack([X1, X2, X3]) # Putting data together
	Y = np.array([0]*N_per_class + [1]*N_per_class + [2]*N_per_class) # Constructing prediction values
	N = N_per_class * K # Calculating the total number of samples

	plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
	plt.show()

	nn = SimpleNeuralNetwork([2, 3, 3])
	costs = nn.train(X, Y, 100000, 0.001)
	y_pred = nn.predict(X)

	plt.plot(costs)
	plt.show()

if __name__ == "__main__":
	main()