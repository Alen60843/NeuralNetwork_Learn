import numpy as np


def sigmoid(x):
    # Output: 1/(1+e^-x)
    x = np.clip(x, -500, 500)  # Clip to avoid overflow
    return 1 / (1+np.exp(-x))

def derivative_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

# Implement a neuron
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def feedforward(self, inputs):
        # Weighted sum of inputs + bias
        total = np.dot(self.weights, inputs) + self.bias
        # Apply activation function
        return sigmoid(total)
    
# MSE LOSS
def mse_loss(y_true, y_pre):
    return np.mean((y_true - y_pre) **2)



class OurNeuralNetwork:
    '''
    A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1,h2)
    - an output layer with 1 neuron (o1)
    Each neuron has the same weghts and bias
    - w = [0,1]
    - b = 0
    '''
    def __init__ (self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Bias
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()   
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 20000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Do a feedforward pass
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 *h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                # Calculate the partial derivatives
                d_L_d_ypred = -2 * (y_true - o1)
                # Neuron o1
                d_ypred_d_w5 = h1 * derivative_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * derivative_sigmoid(sum_o1)
                d_ypred_d_b3 = derivative_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * derivative_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * derivative_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * derivative_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * derivative_sigmoid(sum_h1)
                d_h1_d_b1 = derivative_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * derivative_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * derivative_sigmoid(sum_h2)
                d_h2_d_b2 = derivative_sigmoid(sum_h2)

                # Update weights and biases

                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # Calculate total loss at the end of each epoch
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)# This applies self.feedforward to each row (sample) in the 2D array data
                    loss = mse_loss(all_y_trues, y_preds)




def main():
    data = np.array (
                    [[-2,-1],
                     [25,6],
                     [17,4],
                     [-15,-6]
    ])
    all_y_trues = np.array([1, 0, 0, 1])
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)

    # Make predictions
    emily = np.array([-7, -3])
    frank = np.array([20, 2])
    print("Emily: %.3f" % network.feedforward(emily))
    print("Frank: %.3f" % network.feedforward(frank))

if __name__ =='__main__':
    main()