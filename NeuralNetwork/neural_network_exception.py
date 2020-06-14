class NeuralNetworkInputException(Exception) :
	""" Exception raised for errors in NeuralNetwork invalid input.

	Attributes:
		message -- error explanition
	"""

	def __init__(self, message):
		self.message = message
		super().__init__(self.message)

	def __str___(self):
		return f'{self.message}'
