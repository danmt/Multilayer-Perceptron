import numpy as np
import matplotlib.pyplot as plt
from network import Network
from layer import Layer

def ejercicio2():
	opcion = ""
	opcion2 = ""
	print('\nIntroduzca el número de neuronas para la capa oculta: ')
	no = input('')
	print('\nIntroduzca el número de iteraciones permitidas: ')
	it = input('')
	net = Network([2,int(no),1],0.1)
	while opcion == "":
		print('\nSeleccione el conjunto de entrenamiento: \n')
		print(' 1) 500 patrones dados\n')
		print(' 2) 1000 patrones dados\n')
		print(' 3) 2000 patrones dados\n')
		print(' 4) 500 patrones generados\n')
		print(' 5) 1000 patrones generados\n')
		print(' 6) 2000 patrones generados\n')
		opcion = input('')
		if opcion == "1":
			train_set = np.array(np.loadtxt('datosP2EM2017/datos_P2_EM2017_N500.txt'),dtype=np.float)
			print(train_set)
			print(normalized(train_set))
			#net.train(int(it),train_set,0.05)
		elif opcion == "2":
			train_set = np.array(np.loadtxt('datosP2EM2017/datos_P2_EM2017_N1000.txt'),dtype=np.float)
			net.train(int(it),train_set,0.05)
		elif opcion == "3":
			train_set = np.array(np.loadtxt('datosP2EM2017/datos_P2_EM2017_N2000.txt'),dtype=np.float)
			net.train(int(it),train_set,0.05)
		elif opcion == "4":
			train_set = np.array(np.loadtxt('datosP2EM2017/datos500.txt'),dtype=np.float)
			net.train(int(it),train_set,0.05)
		elif opcion == "5":
			train_set = np.array(np.loadtxt('datosP2EM2017/datos1000.txt'),dtype=np.float)
			net.train(int(it),train_set,0.05)
		elif opcion == "6":
			train_set = np.array(np.loadtxt('datosP2EM2017/datos2000.txt'),dtype=np.float)
			net.train(int(it),train_set,0.05)
		else:
			print('\nDebe introducir \'1\', \'2\', \'3\', \'4\', \'5\' o  \'6\'\n')
			opcion = ""


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)