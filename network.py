import numpy as np
from layer import Layer

class Network:
	def __init__(self,dimensions,bias):
		self.numOfLayers = len(dimensions)
		self.layers = []
		self.input_size = dimensions[0]

		for (i,value) in enumerate(dimensions):
			if i + 1 < len(dimensions):
				nextSize = dimensions[i + 1]
			else:
				nextSize = 0

			layer = Layer(i + 1,value,bias,nextSize)
			self.layers.append(layer)

	def train(self,dataset):
		X = dataset[:,:self.input_size]
		Y = dataset[:,self.input_size:]
		last_layer = self.layers[-1]

		for (i,xi) in enumerate(X):
			self.feed_forward(xi)
			error = 0

			for (j,neuron) in enumerate(last_layer.neurons):
				error = error + 0.5 * (Y[j] - neuron.value)**2

			print(error)


	def feed_forward(self,input_vector):
		W = self.layers[0].get_weights()
		self.layers[0].set_values(input_vector);
		input_vector = np.insert(input_vector,0,1)

		for i in xrange(1,self.numOfLayers):
			W = self.layers[i - 1].get_weights()
			x = self.layers[i - 1].get_values()

			for (j,neuron) in enumerate(self.layers[i].neurons):
				vj = np.dot(W.T[j],x)
				neuron.adjust_value(vj)

	def print_network(self):
		print('\n\nNetwork:\n')

		for (i,layer) in enumerate(self.layers):
			layer.print_layer()
			