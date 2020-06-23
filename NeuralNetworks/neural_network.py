import numpy as np 
from neural_network_exception import NeuralNetworkInputException

class NeuralNetwork:
	""" Class, implementing flexible NeuralNetwork. Please note that this is created only with educational purposes.

	Attributes:
		layers_sizes -- List that represents the sizes of every layer in the NN, counting input and output layers as well.
		weights -- list of numpy arrays, containing the weights for every layer in the NN
		biases -- list of numpy arrays, containing the bias terms of every layer in the NN
		activation_functions -- list of activation functions, used in every neuron layer
	"""

	def __init__(self, layers_sizes, activation_functions):
		if len(layers_sizes) != len(activation_functions) - 1:
			raise NeuralNetworkInputException("Length of layers_sizes list must be with 1 greater than activation_functions list !")


		self.weights = []
		self.biases = []
		self.layers_sizes = layers_sizes
		self.activation_functions = activation_functions
		self.initializeWeightsAndBiases()

	def initializeWeightsAndBiases(self):
		for layer_index in range(len(self.layers_sizes) - 1) :
			weight = np.random.randn(self.layers_sizes[layer_index], self.layers_sizes[layer_index + 1])
			bias = np.random.randn(self.layers_sizes[layer_index + 1])

			self.weights.append(weight)
			self.biases.append(bias)

	@staticmethod
	def softmax(a):
		expA = np.exp(a)

		if (len(a.shape)) == 1:
			return expA / expA.sum()

		return expA / expA.sum(axis = 0, keepdims = True)

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
	def activation(func_name):
		if func_name == "softmax":
			return NeuralNetwork.softmax
		elif func_name == "sigmoid":
			return NeuralNetwork.sigmoid
		elif func_name == "tanh":
			return NeuralNetwork.tanh
		elif func_name == "relu":
			return NeuralNetwork.relu
		else:
			raise NeuralNetworkInputException("Invalid activation function name !")

	def feedforward(self, X): # Think of doing it with batches
		Layers_inputs = []
		Layers_outputs = []

		Layers_inputs.append(X)

		for i in range(1, len(self.weights)):
			function = NeuralNetwork.activation(self.activation_functions[i - 1])
			Layers_outputs.append(function(X.dot(self.weights[i - 1]) + self.biases[i - 1]))
			Layers_inputs.append(Layers_outputs[-1].dot(self.weights[i]) + self.biases[i])

		Layers_outputs.append(NeuralNetwork.softmax(Layers_inputs[-1].dot(self.weights[-1]) + self.biases[-1]))
		Y = Layers_outputs[-1]
		return Y

	@staticmethod
	def softmax_derivative(a):
		s = NeuralNetwork.softmax(a)
		if (len(a.shape) == 1):
			s = s.reshape(-1,1)
		return np.diagflat(s) - np.dot(s, s.T)

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
	def derivative(func_name):
		if func_name == "softmax":
			return NeuralNetwork.softmax_derivative
		elif func_name == "sigmoid":
			return NeuralNetwork.sigmoid_derivative
		elif func_name == "tanh":
			return NeuralNetwork.tanh_derivative
		elif func_name == "relu":
			return NeuralNetwork.relu_derivative
		else:
			raise NeuralNetworkInputException("Invalid activation function name !")

	def backpropagation(self, T, Y):
		deltas = []
		deltas.append(T - Y)

		for i in range(1, len(self.weights)):
			Z_deriv = NeuralNetwork.derivative(self.activation_functions[len(self.activation_functions) - i])
			deltas.append(deltas[i - 1].dot(self.weights[i - 1].T) * Z_deriv)

		dJ = [] # Array to construct the derivatives wrt to every weights and biases layers

def main():
	print("Hi, here is my first ANN from scratch !")

if __name__ == "__main__":
	main()