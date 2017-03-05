# coding=utf-8
from network import Network
from layer import Layer
import numpy as np

def main():
	net = Network([2,4,1],0.1)
	train_set = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
	
	test_x = np.array([[0,0]])
	test_y = np.array([[0]])
	
	net.print_network()
	#print(error)
	net.train(10000,train_set,0.05)
	output = net.predict(test_x)

	print("\nOutput:\n")
	print("desired="),
	print(test_y)
	print("output="),
	print(output)

	net.print_network()
	#net.print_network()


if __name__ == "__main__":
    main()