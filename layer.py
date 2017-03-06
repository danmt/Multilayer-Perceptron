import numpy as np
from neuron import Neuron

class Layer:
	def __init__(self,num,size,bias):
		self.neurons = []
		self.num = num
		self.size = size
		self.bias = bias
		self.in_synapses = []

		for i in range(0,size):
			self.neurons.append(Neuron(i + 1,bias))

	def get_values(self):
		values = np.array([1])
		
		for (i,neuron) in enumerate(self.neurons):
			values = np.concatenate((values,[neuron.value]),axis=0)

		return values

	def get_neurons_as_array(self):
		new_y = []

		for neur in self.neurons:
			new_y.append(neur.value)

		return np.array([new_y])

	def get_gradients_as_array(self):
		new_y = []

		for neur in self.neurons:
			new_y.append(neur.gradient)

		return np.array(new_y)

	def get_weights(self,num):
		new_weigths = []

		for x in range(0,len(self.in_synapses)):
			new_weigths.append(self.in_synapses[x][num])

		return np.array(new_weigths)

	def set_values(self,values):
		for (i,neuron) in enumerate(self.neurons):
			neuron.adjust_value(values[i])

	def set_synapses(self,synapses):
		self.in_synapses = synapses

	def print_layer(self):
		print('\nLayer ' + str(self.num))

		for (i,neuron) in enumerate(self.neurons):
			neuron.print_neuron()		

