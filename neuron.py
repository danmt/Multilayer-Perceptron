import numpy as np

class Neuron:
	def __init__(self,num,synapses,bias):
		self.num = num
		self.bias = bias
		self.value = np.random.rand()
		
		if synapses == 0:
			self.synapses = np.array([])
		else:
			self.synapses = np.random.rand(synapses)

	def adjust_value(self,value):
		self.value = value

	def print_neuron(self):
		print('Neuron ' + str(self.num)),
		print(' value = ' + str(self.value))

		print('Synapses: ')
		print(self.synapses)