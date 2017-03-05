# coding=utf-8
from network import Network
from layer import Layer
import numpy as np

def main():

	opcion = ""
	opcion2 = ""
	print('Redes Neuronales: Clasificación de Patrones.\n\n')
	print('El primer conjunto de datos (1) corresponde a la clasificación\n')
	print('de puntos (x,y) según la región a la que pertenecen, mientras que el\n')
	print('el segundo conjunto de datos (2) corresponde a clasificación de\n')
	print('Iris según su clase.\n\n')
	while opcion == "":
		print('Seleccione el conjuto de datos de su preferencia:\n')
		opcion = input('')
		if opcion == "1":
			#ejercicio_1()
		elif opcion == "2":
			while opcion2 == "":
				print('\n')
				print('Seleccione la clasificación deseada:\n')
				print('1) Iris Setosa \n')
				print('2) Iris Setosa, Iris Versicolor e Iris Virginica \n')
				opcion2 = input('')
				if opcion2 == "1" or opcion2 == "2":
					#ejercicio_2(opcion2)
				else:
					print('Debe introducir \'1\' o \'1\' \n')
					opcion2 = ""
		else:
			print('\nDebe introducir \'1\' o \'2\' \n')
			opcion = ""


	net = Network([2,3,1],0.25)
	dataset = np.array([[1,2,0],[0.4,0.7,0],[0.7,0.8,1],[0.4,0.1,1]])
	datas= np.matrix(np.loadtxt('nombre.txt'), dtype=np.float128)

	net.print_network()
	net.train(dataset)
	net.print_network()


if __name__ == "__main__":
    main()