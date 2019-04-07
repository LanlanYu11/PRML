# -*- coding=utf-8 -*-
import numpy as np

class BP(object):
 def __init__(self, input_dim, hidden_dim, lr):
  self.W1 = np.random.rand(hidden_dim, input_dim)    # j*i
  self.bias1 = np.random.rand(hidden_dim, 1)   # j*1
  self.W2 = np.random.rand(1, hidden_dim)   # 1*j
  self.bias2 = np.random.rand(1, 1)   # 1*1
  self.lr = lr
 
 def getWeight(self):
  return self.W1, self.W2
 
 def getbias(self):
  return self.bias1, self.bias2
 
 def getLoss(self):
  return self.loss
  
 def __action(self, a):   # kernel
  return 1.0/(1 + np.exp(-a))
 
 def __daction(self, a):    # derivative of kernel
  return a - np.power(a, 2)

# @ matrix product, np.multiply 对应值相乘 == *
 def __forward(self, X):
  self.a1 = self.W1 @ X + self.bias1   # j*k
  #print("j*k", self.a1.shape)
  self.h1 = self.__action(self.a1)   # j*k
  #print("j*k", self.h1.shape)
  y_pred = self.W2 @ self.h1 + self.bias2   # regression  1*k
  #print("1*k", y_pred.shape)
  return y_pred
 
 def __back(self, X):   # X is i*k, label and y_pred is 1*k
  dim, k = X.shape
  # update second layer
  self.W2 = self.W2 - self.lr*(self.loss @ self.h1.T)  # 1*j = 1*k X (j*k).T
  #print("", self.W2.shape)
  self.bias2 = self.bias2 - self.lr*(self.loss @ np.ones((k, 1)))
  #print(self.W2.shape, self.bias2.shape)
  
  # update first layer
  delta1 = self.W2.T * (self.loss * self.__daction(self.h1))  # (j*1) () , j is the number of hidden layer
  #print("j*k", delta1.shape)   # j*k
  self.W1 = self.W1 - self.lr*(delta1 @ X.T)
  self.bias1 = self.bias1 - self.lr*(delta1 @ np.ones((k, 1)))
  #print(self.W1.shape, self.bias1.shape)

 def train(self, epoch_size, X, label):
  for epoch in np.arange(epoch_size):
   y_pred = self.__forward(X)
   self.loss = y_pred - label   # 1*k , k is the number of sample
   self.__back(X)
   print(epoch," loss: ", self.loss @ self.loss.T)
 
 def test(self, X):
  y_pred = self.__forward(X)
  return y_pred
  
 def evaluate(self, y, label):
  return 0.5 * 1.0/len(y) * np.power((y-label)@(y-label).T, 0.5)   # 1*k X k*1

if __name__ == '__main__':
 
 train_file = '000011_train.txt'
 test_file = '000011_test.txt'
 train_data = np.array(np.loadtxt(train_file))
 print(train_data.shape)
 input_num, input_dim = train_data.shape
 train_x = train_data[:, 1::].T
 train_y = train_data[:, 0].T
 test_data = np.array(np.loadtxt(test_file))
 test_x = test_data[:, 1::].T
 test_y = test_data[:, 0].T
 
 epoch_size = 500
 hidden_dim = int(input_dim/2)
 print(input_dim, hidden_dim)
 # try 
 '''
 input_dim = 4
 hidden_dim = 1
 train_x = np.array([[1, 0, 2, 1],[0, 3, 1, 1],[3, 2, 2, 3]])
 train_y = np.array([6, 8, 10, 8])'''
 bp = BP(input_dim-1, hidden_dim, 0.00001)
 # train nn
 bp.train(epoch_size, train_x, train_y)
 y_pred = bp.test(test_x)
 
 # print("train loss: ", bp.getLoss() @ bp.getLoss.T)
 print("test loss: ", bp.evaluate(y_pred, test_y))