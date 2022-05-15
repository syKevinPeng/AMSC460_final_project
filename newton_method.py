'''
Newton's Method Implementation
'''

import numpy as np
from sklearn.metrics import accuracy_score
from random import shuffle

class NewtonMethod():
    def __init__(self):
        self.losses = []
        self.train_accuracies = []
        self.weights = None
    
    # Defining sigmoid function specifically for newton's method
    def sigmoid(self, x):                                                                                                   
        return 1.0 / (1.0 + np.exp(-x))      
     
    # defining hessian matrix
    def hessian(self, sig_out, data):                                                          
        sig_der = np.diag(np.multiply(sig_out, np.subtract(1, sig_out)))
        hess = np.matmul(np.matmul(data,sig_der), np.transpose(data))                             
        return hess
    
    def binary_cross_entropy_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def shuffle_dataset(self, x, y):
        y = np.expand_dims(y, axis=1)
        dataset = np.concatenate((x, y), axis = 1)
        np.random.shuffle(dataset)
        x = dataset[:, 0:-1]
        y = dataset[:, -1]
        return x,y

    def fit(self, x, y, epoch = 100,  verbose = False):
        params_num = x.shape[1]
        data_length = x.shape[0]
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=params_num)                                                             
        for i in range(epoch):
            # shuffle dataset
            x, y = self.shuffle_dataset(x, y)

            sig_out = np.zeros(data_length)
            diff = np.zeros(data_length)
            gradient = np.zeros((params_num, 1))
            data = np.zeros((params_num, data_length))
            # row iterator loop
            for j in range(data_length):
                sig_out[j] = self.sigmoid(np.dot(x[j], self.weights))
                diff[j] = sig_out[j] - y[j]
                data[:,j] = x[j].transpose()
                gradient[:, 0] = gradient[:,0] + np.multiply(x[j].transpose(), diff[j])

            # compute Hessian
            hess = self.hessian(sig_out, data)
            inv_hess = np.linalg.inv(hess)

            # do the weight update
            self.weights = self.weights  - np.squeeze(np.matmul(inv_hess, gradient))

            pred = self.sigmoid(np.matmul(self.weights, x.transpose()))
            loss = self.binary_cross_entropy_loss(y, pred) 
            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            acc = accuracy_score(y, pred_to_class)
            self.losses.append(loss)
            self.train_accuracies.append(acc)
            if verbose: print(f'Epoch {i}: loss:{loss}; acc:{acc}\n')

        return [self.weights], [self.train_accuracies, self.losses]

    def test(self, test_x, test_y):
        output = np.matmul(self.weights, test_x.transpose()) 
        pred = self.sigmoid(output)
        pred_to_class = [1 if p > 0.5 else 0 for p in pred]
        test_acc = accuracy_score(test_y, pred_to_class)
        return test_acc