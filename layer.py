import numpy as np
from neuron import Neuron

class Layer:
	def __init__(self,num,size,bias,nextLayerSize):
		self.neurons = []
		self.nextLayerSize = nextLayerSize
		self.num = num
		self.size = size
		self.bias = bias

		if not nextLayerSize:
			nextLayerSize = 0

		for i in xrange(0,size):
			self.neurons.append(Neuron(i + 1,nextLayerSize,bias))

	def get_weights(self):
		biases = np.ones(self.nextLayerSize)
		weights = np.array([biases*self.bias])

		for (i,neuron) in enumerate(self.neurons):
			weights = np.concatenate((weights,[neuron.synapses]),axis=0)

		return weights

	def get_values(self):
		values = np.array([1])

		for (i,neuron) in enumerate(self.neurons):
			values = np.concatenate((values,[neuron.value]),axis=0)

		return values

	def set_values(self,values):
		for (i,neuron) in enumerate(self.neurons):
			neuron.adjust_value(values[i])
			#values = np.concatenate((values,[neuron.value]),axis=0)

		#return values

	def print_layer(self):
		print('\nLayer ' + str(self.num))

		for (i,neuron) in enumerate(self.neurons):
			neuron.print_neuron()

