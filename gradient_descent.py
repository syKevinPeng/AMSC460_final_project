'''
Gradient Descent Method Implementation
'''
import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression():
    def __init__(self, learning_rate):
        self.losses = []
        self.train_accuracies = []
        self.l_rate = learning_rate

    # define sigmoid function for gradient descent method
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - self.l_rate * error_w
        self.bias = self.bias - self.l_rate * error_b

    def fit(self, x, y, epochs, verbose):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        early_stop_tol = 1e-5
        early_stop_buffer = []

        for i in range(epochs):
            output = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self.sigmoid(output)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            acc = accuracy_score(y, pred_to_class)
            self.train_accuracies.append(acc)
            self.losses.append(loss)

            if verbose: print(f'Epoch {i}: loss:{loss}; acc:{acc}\n')
            # implement early stop
            if len(early_stop_buffer) > 5:
                early_stop_buffer.pop(0)
            early_stop_buffer.append(acc)
            if len(early_stop_buffer) == 5 and (max(early_stop_buffer) - min(early_stop_buffer) < early_stop_tol):
                break # exit for loop
        return [self.weights, self.bias], [self.train_accuracies, self.losses]
    
    def test(self, test_x, test_y):
        output = np.matmul(self.weights, test_x.transpose()) + self.bias
        pred = self.sigmoid(output)
        pred_to_class = [1 if p > 0.5 else 0 for p in pred]
        test_acc = accuracy_score(test_y, pred_to_class)
        return test_acc


