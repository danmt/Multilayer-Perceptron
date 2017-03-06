import math
import random
import string
import numpy as np

class MLP:

  def __init__(self, ni, nh, no):
    # number of nodes for each layer
    self.ni = ni + 1 # +1 for bias
    self.nh = nh
    self.no = no
    
    # node-activations
    self.ai = []
    self.ah = [] 
    self.ao = []

    # fill with ones
    self.ai = [1.0]*self.ni
    self.ah = [1.0]*self.nh
    self.ao = [1.0]*self.no

    # create node weight matrices with random fill
    self.wi = np.random.uniform(-0.2,0.2,(self.ni, self.nh))
    self.wo = np.random.uniform(-2.0,2.0,(self.nh, self.no))

    # create last change in weights matrices for momentum
    self.ci = np.zeros((self.ni, self.nh), dtype=float)
    self.co = np.zeros((self.nh, self.no), dtype=float)
    
  def fordwardpropagation(self, inputs):
  
    for i in range(self.ni-1):
      self.ai[i] = inputs[i]
    
    # propagation from input to hidden
    for j in range(self.nh):
      sum = 0.0
      for i in range(self.ni):
        sum +=( self.ai[i] * self.wi[i][j])
      self.ah[j] = sigmoid(sum)
    
    # propagation from hidden to output
    for k in range(self.no):
      sum = 0.0
      for j in range(self.nh):        
        sum +=( self.ah[j] * self.wo[j][k] )
      self.ao[k] = sigmoid(sum)
    
    # return result
    return self.ao

  
  def backpropagation(self, targets, N, M):

    # calculate deltas of layers
    output_deltas = np.zeros((self.no,1), dtype=float)

    for k in range(self.no):
      error = targets[k] - self.ao[k]
      output_deltas[k] =  error * dsigmoid(self.ao[k]) 
   
    # update weights
    for j in range(self.nh):
      for k in range(self.no):
        change = output_deltas[k] * self.ah[j]
        self.wo[j][k] += N*change + M*self.co[j][k]
        self.co[j][k] = change

    # calculate deltas of hidden layer
    hidden_deltas = np.zeros((self.nh,1), dtype=float)

    for j in range(self.nh):
      error = 0.0
      for k in range(self.no):
        error += output_deltas[k] * self.wo[j][k]
      hidden_deltas[j] = error * dsigmoid(self.ah[j])
    
    # update weights
    for i in range (self.ni):
      for j in range (self.nh):
        change = hidden_deltas[j] * self.ai[i]
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change
        
    # calculate error
    error = 0.0
    for k in range(len(targets)):
      error += 0.5 * (targets[k]-self.ao[k])**2
    return error
          
  def predict(self, patterns):
    for p in patterns:
      inputs = p[:-1]
      print ('Inputs:', p[-1], '-->', self.fordwardpropagation(inputs), '\tTarget', p[-1])
  
  def train(self, patterns, iterations =1000, N=0.05, M=0.1):
    error_acc = 0
    i = 0
    epsilon = 0.12
    converge = False
    while(i < iterations and (not converge)):
      for i in range(iterations):
        errors =[]
        for p in patterns:
          inputs = p[:-1]
          targets = [p[-1]]
          self.fordwardpropagation(inputs)
          error = self.backpropagation(targets, N, M)
          errors.append(error)

        errors = np.array(errors)
        errorN = (1/len(patterns))*np.sum(errors**2) # iteration error
        error_acc = error_acc + errorN
        if(errorN <= epsilon):
          converge = True
        i += 1

    error_acc = error_acc / i
    print(error_acc)

def sigmoid(x):
  return math.tanh(x)
  
def dsigmoid(y):
  return 1 - y**2

