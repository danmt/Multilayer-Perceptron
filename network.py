import numpy as np
from layer import Layer
from scipy.special import expit

def sigmoid_prime(x):
    return expit(x)*(1.0-expit(x))

def get_synapses(rows,cols,bias):
	synapses_matrix = np.random.normal(0,0.1,(rows,cols))
	synapses_matrix = synapses_matrix.T
	synapses_matrix = np.insert(synapses_matrix,0,1 * bias,axis=1)
	return synapses_matrix

class Network:
	def __init__(self,dimensions,bias):
		self.numOfLayers = len(dimensions)
		self.layers = []
		self.input_size = dimensions[0]
		self.output_size = dimensions[-1]

		for (i,value) in enumerate(dimensions):
			layer = Layer(i + 1,value,bias)
			self.layers.append(layer)

			if i != 0:

				synapses = get_synapses(dimensions[i-1],dimensions[i],bias)
				self.layers[i].set_synapses(synapses)

	def train(self,epochs,dataset,learning_rate):
		X = dataset[:,:self.input_size]
		Y = dataset[:,self.input_size:]
		err_acc = 0

		for epoch in range(1,epochs+1):
			errors = []
			for (i,xi) in enumerate(X):
				self.feed_forward(xi)
				err = self.backward_pass(Y[i],learning_rate)
				errors.append(err)

			errors = np.array(errors)
			errorN = (1/len(X))*np.sum(errors**2) #Error cuadratico medio iteracion n
			err_acc = err_acc + errorN



			if (epoch % 100 == 0):
				print('\nEpoch #' + str(epoch))
				print('avg err = '),
				print("")
				print(errorN)

		err_acc = err_acc / epochs
		print(err_acc)


	def feed_forward(self,input_vector):
		self.layers[0].set_values(input_vector)

		for i in range(1,self.numOfLayers):
			W = self.layers[i].in_synapses
			x = self.layers[i - 1].get_values()

			value = np.dot(W,x.T)

			for (j,neuron) in enumerate(self.layers[i].neurons):
				neuron.adjust_value(expit(value[j]))

	def backward_pass(self,Y,learning_rate):
		layers = self.layers

		for (n,layer) in enumerate(layers[::-1]):
			if n + 1 == len(self.layers):
				continue

			if n > 0:
				past_w = w
				past_layer = self.layers[layer.num]

			w = layer.in_synapses

			for (i,neuron) in enumerate(layer.neurons):
				if n == 0: #CASO CAPA SALIDA
					error_out = Y[i] - neuron.value
					#print("esperado: " + str(Y[i]) + " neurona: " + str(neuron.value))
					#print(error_out)
				else: #CASO CAPAS OCULTA
					err_array = []
					for (j,neur) in enumerate(past_layer.neurons):
						err_array.append(neur.gradient * past_w[j][i + 1])

					error_out = np.sum(err_array)

				delta = sigmoid_prime(neuron.value)
				local_gradient = error_out * delta
				neuron.set_gradient(local_gradient)

				for (j,neur) in enumerate(layers[layer.num - 2].neurons):
					if (j == 0):
						w[i][j] = w[i][j] - (learning_rate * local_gradient)
					else:
						w[i][j] = w[i][j] + (learning_rate * neur.value * local_gradient)

		output = self.layers[self.numOfLayers - 1].get_neurons_as_array()				
		error = Y - output
		return 0.5*np.sum((error)**2)

	def predict(self,x,y):
		self.layers[0].set_values(x)

		for i in range(1,self.numOfLayers):
			W = self.layers[i].in_synapses
			x = self.layers[i - 1].get_values()

			value = np.dot(W,x.T)

			for (j,neuron) in enumerate(self.layers[i].neurons):
				neuron.adjust_value(expit(value[j]))

		output = self.layers[self.numOfLayers - 1].get_neurons_as_array()
		output = np.sum(output-y)

		if output <= 0:
			return 1
		else:
			return 0

	def print_network(self):
		print('\n\nNetwork:\n')

		for (i,layer) in enumerate(self.layers):
			layer.print_layer()
			if hasattr(layer,'in_synapses'):
				print(layer.in_synapses)
			