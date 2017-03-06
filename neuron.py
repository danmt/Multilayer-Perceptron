import numpy as np

class Neuron:
	def __init__(self,num,bias):
		self.num = num
		self.bias = bias
		self.gradient = None
		self.value = np.random.normal(0,0.1)

	def adjust_value(self,value):
		self.value = value

	def set_gradient(self,gradient):
		self.gradient = gradient

	def print_neuron(self):
		print('Neuron ' + str(self.num)),
		print(' value = ' + str(self.value)),
		print(' gradient = ' + str(self.gradient))