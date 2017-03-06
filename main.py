# coding=utf-8
from network import Network
from layer import Layer
import numpy as np

def main():
	net = Network([2,3,1],0.1)
	train_set = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
	
	net.train(10000,train_set,0.01)
	net.print_network()
	

	test1_x = np.array([0,0])
	test1_y = np.array([0])

	output1 = net.predict(test1_x,test1_y)

	print("\nOutput:\n")
	print("desired="),
	print(test1_y)
	print("output="),
	print(output1)

	test2_x = np.array([1,0])
	test2_y = np.array([1])

	output2 = net.predict(test2_x,test2_y)

	print("\nOutput:\n")
	print("desired="),
	print(test2_y)
	print("output="),
	print(output2)

	test3_x = np.array([0,1])
	test3_y = np.array([1])

	output3 = net.predict(test3_x,test3_y)

	print("\nOutput:\n")
	print("desired="),
	print(test3_y)
	print("output="),
	print(output3)

	test4_x = np.array([1,1])
	test4_y = np.array([0])

	output4 = net.predict(test4_x,test4_y)

	print("\nOutput:\n")
	print("desired="),
	print(test4_y)
	print("output="),
	print(output4)
	#net.print_network()
	#net.print_network()


if __name__ == "__main__":
    main()