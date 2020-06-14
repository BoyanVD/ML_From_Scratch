import numpy as np 
from neural_network_exception import NeuralNetworkInputException

class NeuralNetwork :
	""" Class that implements simple feedforward Neural Network from scratch, using pure python and numpy.

	Attributes:
		layers_sizes -- List that represents the sizes of every layer in the NN, counting input and output layers as well.
		layers_activation_functions -- List, representing the activation functions on every layer.
		weights -- list of numpy arrays, containing the weights for every layer in the NN
		biases -- list of numpy arrays, containing the bias terms of every layer in the NN
	"""


	def __init__(self, layers_sizes, layers_activation_functions) :
		if len(layers_sizes) != (len(layers_activation_functions) + 1) :
			raise NeuralNetworkException("Invalid layers sizes and layers activation functions lists sizes !")

		self.weights = []
		self.biases = []
		self.layers_sizes = layers_sizes
		self.layers_activation_functions = layers_activation_functions

		for layer_index in range(len(self.layers_sizes) - 1) :
			weight = np.random.randn(self.layers_sizes[layer_index], self.layers_sizes[layer_index + 1])
			bias = np.random.randn(self.layers_sizes[layer_index + 1])

			self.weights.append(weight)
			self.biases.append(bias)

	def feed_forward(self, X):
		Z = np.copy(X)
		A_values = [] # Values before activation functions in neurons (Neurons input)
		Z_values = [Z] # Values after activation functions in neurons (Neurons output)

		for index in range(len(self.weights)):
			function = NeuralNetwork.activation(self.layers_activation_functions[index])
			layer_weights = self.weights[index]
			layer_biases = self.biases[index]

			A = Z.dot(layer_weights) + layer_biases
			A_values.append(A)

			Z = function(A)
			Z_values.append(Z)

		return A_values, Z_values

	@staticmethod
	def softmax(a):
		expA = np.exp(a)
		return expA / expA.sum(axis = 1, keepdims = True)

	@staticmethod
	def sigmoid(a):
		return 1 / (1 + np.exp(-a))

	@staticmethod
	def tanh(a):
		return np.tanh(a)

	@staticmethod
	def relu(x):
		y = np.copy(x)
		y[y < 0] = 0

		return y

	@staticmethod
	def sigmoid_derivative(a):
		return NeuralNetwork.sigmoid(a) * (1 - NeuralNetwork.sigmoid(a))

	@staticmethod
	def tanh_derivative(a):
		return 1 - np.tanh(a)**2

	@staticmethod
	def relu_derivative(a):
		y = np.copy(x)
		y[y >= 0] = 1
		y[y < 0] = 0
		return y

	@staticmethod
	def activation(name):
		if name == "sigmoid":
			return NeuralNetwork.sigmoid
		elif name == "tanh":
			return NeuralNetwork.tanh
		elif name == "relu":
			return NeuralNetwork.relu
		elif name == "softmax":
			return NeuralNetwork.softmax
		else:
			raise NeuralNetworkException("Invalid activation function name !")

	@staticmethod
	def derivative(name):
		if name == "sigmoid":
			return NeuralNetwork.sigmoid_derivative
		elif name == "tanh":
			return NeuralNetwork.tanh_derivative
		elif name == "relu":
			return NeuralNetwork.relu_derivative
		else:
			raise NeuralNetworkException("Invalid activation function name !")