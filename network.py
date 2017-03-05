import numpy as np
from layer import Layer
from scipy.special import expit

def sigmoid_prime(x):
    return expit(x)*(1.0-expit(x))

def get_synapses(rows,cols,bias):
	bias_vector = np.ones(rows)
	synapses_matrix = np.random.normal(0,0.1,(rows,cols))
	return np.insert(synapses_matrix,0,1 * bias,axis=1)

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
				synapses = get_synapses(dimensions[i],dimensions[i-1],bias)
				self.layers[i].set_synapses(synapses)

	def train(self,epochs,dataset,learning_rate):
		X = dataset[:,:self.input_size]
		Y = dataset[:,self.input_size:]
		err_acc = 0

		for epoch in xrange(1,epochs+1):
			errors = []

			for (i,xi) in enumerate(X):
				self.feed_forward(xi)
				err = self.backward_pass(Y[i],learning_rate)
				errors.append(err)

			errors = np.array(errors)
			errorN = 0.5*np.sum(errors**2) #Error cuadratico medio iteracion n
			err_acc = err_acc + errorN

			if (epoch % 1000 == 0):
				print('\nEpoch #' + str(epoch))
				print('avg err = '),
				print(errorN)


		err_acc = err_acc / epochs
		print(err_acc)


	def feed_forward(self,input_vector):
		input_vector = np.insert(input_vector,0,1)		
		self.layers[0].set_values(input_vector)

		for i in xrange(1,self.numOfLayers):
			W = self.layers[i].in_synapses
		 	x = self.layers[i - 1].get_values()

			for (j,neuron) in enumerate(self.layers[i].neurons):
				vj = np.dot(W[j],x)
				neuron.adjust_value(expit(vj))

	def backward_pass(self,Y,learning_rate):
		layers = self.layers
		
		for (n,layer) in enumerate(layers[::-1]):
			if n + 1 == len(self.layers):
				continue

			w = layer.in_synapses

			for (i,neuron) in enumerate(layer.neurons):
				if n == 0: #CASO CAPA SALIDA
					error_out = Y[i] - neuron.value
				else: #CASO CAPAS OCULTA
					out_layer = self.layers[layer.num]
					y = out_layer.get_gradients_as_array()
					weights = out_layer.get_weights(neuron.num)
					error_out = np.dot(weights,y)

				delta = sigmoid_prime(neuron.value)
				local_gradient = error_out * delta
				neuron.set_gradient(local_gradient)

				#Actualizo el sesgo
				w[i][0] = w[i][0] - (learning_rate * local_gradient)

				for (j,neur) in enumerate(layers[layer.num - 2].neurons):
					w[i][j] = w[i][j] + (learning_rate * neur.value * local_gradient)

		output = self.layers[self.numOfLayers - 1].get_neurons_as_array()				
		error = Y - output
		return 0.5*np.sum((error)**2)

	def predict(self,x):
		x = np.insert(x,0,1)		
		self.layers[0].set_values(x)

		for i in xrange(1,self.numOfLayers):
			W = self.layers[i].in_synapses
		 	x = self.layers[i - 1].get_values()

			for (j,neuron) in enumerate(self.layers[i].neurons):
				vj = np.dot(W[j],x)
				neuron.adjust_value(expit(vj))

		output = self.layers[self.numOfLayers - 1].get_neurons_as_array()
		return expit(np.sum(output))

	def print_network(self):
		print('\n\nNetwork:\n')

		for (i,layer) in enumerate(self.layers):
			layer.print_layer()
			if hasattr(layer,'in_synapses'):
				print(layer.in_synapses)
			