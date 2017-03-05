# coding=utf-8
from network import Network
from layer import Layer
import numpy as np

def main():
	net = Network([2,3,1],0.25)
	dataset = np.array([[1,2,0],[0.4,0.7,0],[0.7,0.8,1],[0.4,0.1,1]])

	net.print_network()
	net.train(dataset)
	net.print_network()


if __name__ == "__main__":
    main()