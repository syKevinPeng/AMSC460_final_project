'''
Newton's Method Implementation
'''

import numpy as np

class NewtonMethod():
    def __init__(self):
        self.losses = []
        self.train_accuracies = []

    # Defining sigmoid function specifically for newton's method
    def sigmoid(self, x, Θ_1, Θ_2):                                                        
        z = (Θ_1*x + Θ_2).astype("float_")                                              
        return 1.0 / (1.0 + np.exp(-z))      

    # Defining log likelihood function using sigmoid
    def log_likelihood(self, x, y, Θ_1, Θ_2):                                                                
        sigmoid_probs = self.sigmoid(x, Θ_1, Θ_2)                                        
        return np.sum(y * np.log(sigmoid_probs) + (1 - y) * np.log(1 - sigmoid_probs)) 

    # Defining gradient for the above function
    def gradient(self, x, y, Θ_1, Θ_2):                                                         
        sigmoid_probs = self.sigmoid(x, Θ_1, Θ_2)                                        
        return np.array([[np.sum((y - sigmoid_probs) * x), np.sum((y - sigmoid_probs) * 1)]])       

    # defining hessian matrix
    def hessian(self, x, y, Θ_1, Θ_2):                                                          
        sigmoid_probs = self.sigmoid(x, Θ_1, Θ_2)                                        
        d1 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x * x)                  
        d2 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x * 1)                  
        d3 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * 1 * 1)                  
        hessian = np.array([[d1, d2],[d2, d3]])                                           
        return hessian

    def newtons_method(self, x, y, tol = .0000000001, max_step = 15):                                                             
        """
        :param x : Input Data
        :param y : Label
        :returns: two logestic regression parameters
        """

        # Initialize parameters                                                                   
        theta = 15.1                                                                     
        intercept = -0.4                                                           
        delta = 100 # initilize delta to a relateively large term                                                              
        l = self.log_likelihood(x, y, theta, intercept)                                                                 

        # iter steps                                                                                                           
        iter = 0                                                                           
        while abs(delta) > tol and iter < max_step:                                       
            iter += 1                                                                      
            grad = self.gradient(x, y, theta, intercept)                                                      
            hess = self.hessian(x, y, theta, intercept)                                                                                               

            update_matrix = np.linalg.inv(hess)@grad.T                                                             
            delta_1 = update_matrix[0][0]                                                              
            delta_2 = update_matrix[1][0]                                                              
                                                                                        
            # Update parmeters                                                    
            theta += delta_1                                                                 
            intercept += delta_2                                                                 
                                                                                        
            # Update weights                                   
            l_new = self.log_likelihood(x, y, theta, intercept)                                                      
            delta = l - l_new                                                           
            l = l_new                                                                
        return np.array([theta, intercept]) 