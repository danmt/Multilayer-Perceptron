import numpy as np
from layer import Layer
from scipy.special import expit

def sigmoid_prime(x):
    return expit(x)*(1.0-expit(x))

def get_synapses(rows,cols,bias):
	#bias_vector = np.ones(rows)
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

		for epoch in xrange(1,epochs+1):
			errors = []

			for (i,xi) in enumerate(X):
				self.feed_forward(xi)
				err = self.backward_pass(Y[i],learning_rate)
				errors.append(err)

			errors = np.array(errors)
			errorN = 0.5*np.sum(errors**2) #Error cuadratico medio iteracion n
			err_acc = err_acc + errorN

			if (epoch % 100 == 0):
				print('\nEpoch #' + str(epoch))
				print('avg err = '),
				print(errorN)


		err_acc = err_acc / epochs
		#print(err_acc)


	def feed_forward(self,input_vector):
		self.layers[0].set_values(input_vector)

		#print('\nFeed forward')

		for i in xrange(1,self.numOfLayers):
			W = self.layers[i].in_synapses
		 	x = self.layers[i - 1].get_values()

		 	value = np.dot(W,x.T)

		 	# print('W='),
		 	# print(W)
		 	# print('x='),
		 	# print(x)
		 	# print('value='),
		 	# print(value)

			for (j,neuron) in enumerate(self.layers[i].neurons):
				neuron.adjust_value(expit(value[j]))

	def backward_pass(self,Y,learning_rate):
		layers = self.layers

		#print('\nBack pass')
		
		for (n,layer) in enumerate(layers[::-1]):
			if n + 1 == len(self.layers):
				continue

			if n > 0:
				past_w = w
				past_layer = self.layers[layer.num]


			w = layer.in_synapses

			# print('w='),
			# print(w)

			# print('Y='),
			# print(Y)

			for (i,neuron) in enumerate(layer.neurons):
				# print('Iteracion #' + str(i))
				# print('neuron='),
				# print(neuron.value)
				if n == 0: #CASO CAPA SALIDA
					error_out = Y[i] - neuron.value
				else: #CASO CAPAS OCULTA
					err_array = []
					#print('LA PRUEBA\n\n\n\n')
					for (j,neur) in enumerate(past_layer.neurons):
						# print(neur.num)
						# print(neur.gradient)
						# print(past_w[j])
						# print('i='),
						# print(i)
						# print(past_w[j][i + 1])
						#print(w[j][i+1])
						err_array.append(neur.gradient * past_w[j][i + 1])

					error_out = np.sum(err_array)
					#past_w[i+1]		

					#out_layer = self.layers[layer.num]
					#y = out_layer.get_gradients_as_array()
					#weights = out_layer.get_weights(neuron.num)
					#error_out = np.dot(weights,y)

				
				# print('error_out='),
				# print(error_out)


				delta = sigmoid_prime(neuron.value)
				local_gradient = error_out * delta
				neuron.set_gradient(local_gradient)

				# print('delta='),
				# print(delta)

				# print('local_gradient='),
				# print(local_gradient)

				#Actualizo el sesgo
				#w[i][0] = w[i][0] - (learning_rate * local_gradient)

				for (j,neur) in enumerate(layers[layer.num - 2].neurons):
					if (j == 0):
						w[i][j] = w[i][j] - (learning_rate * local_gradient)
					else:
						w[i][j] = w[i][j] + (learning_rate * neur.value * local_gradient)

		output = self.layers[self.numOfLayers - 1].get_neurons_as_array()				
		error = Y - output
		return 0.5*np.sum((error)**2)

	def predict(self,x):
		self.layers[0].set_values(x)

		for i in xrange(1,self.numOfLayers):
			W = self.layers[i].in_synapses
		 	x = self.layers[i - 1].get_values()

		 	value = np.dot(W,x.T)

			for (j,neuron) in enumerate(self.layers[i].neurons):
				neuron.adjust_value(expit(value[j]))

		output = self.layers[self.numOfLayers - 1].get_neurons_as_array()
		return output

	def print_network(self):
		print('\n\nNetwork:\n')

		for (i,layer) in enumerate(self.layers):
			layer.print_layer()
			if hasattr(layer,'in_synapses'):
				print(layer.in_synapses)
			