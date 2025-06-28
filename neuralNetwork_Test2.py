import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class OurNeuralNetwork:
    def __init__(self, input_size=2, hidden_size=6):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weights = {
            "h": np.random.normal(size=(hidden_size, input_size)),  
            "o": np.random.normal(size=(hidden_size,))              
        }
        self.biases = {
            "h": np.random.normal(size=(hidden_size,)),             
            "o": np.random.normal()                                 
        }

    def feedforward(self, x):
        sum_h = np.dot(self.weights["h"], x) + self.biases["h"]  
        h = sigmoid(sum_h)                                       
        sum_o = np.dot(self.weights["o"], h) + self.biases["o"]  
        o = sigmoid(sum_o)
        return o

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 20000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Forward pass 
                sum_h = np.dot(self.weights["h"], x) + self.biases["h"]
                h = sigmoid(sum_h)
                sum_o = np.dot(self.weights["o"], h) + self.biases["o"]
                o = sigmoid(sum_o)

                # Backpropagation
                d_L_d_ypred = -2 * (y_true - o)

                # Output neuron gradients
                d_ypred_d_sum_o = derivative_sigmoid(sum_o)
                d_sum_o_d_w_o = h
                d_sum_o_d_h = self.weights["o"]

                # Hidden layer gradients
                d_h_d_sum_h = derivative_sigmoid(sum_h)
                d_L_d_h = d_L_d_ypred * d_ypred_d_sum_o * d_sum_o_d_h
                d_L_d_sum_h = d_L_d_h * d_h_d_sum_h

                #  Using np.outer to calculate weight gradients for hidden layer
                d_L_d_w_h = np.outer(d_L_d_sum_h, x)  # shape (6, 2)

                d_L_d_b_h = d_L_d_sum_h

                # Update weights and biases
                self.weights["o"] -= learn_rate * d_L_d_ypred * d_ypred_d_sum_o * d_sum_o_d_w_o
                self.biases["o"] -= learn_rate * d_L_d_ypred * d_ypred_d_sum_o
                self.weights["h"] -= learn_rate * d_L_d_w_h
                self.biases["h"] -= learn_rate * d_L_d_b_h

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                # Uncomment to log:
                # print(f"Epoch {epoch}, Loss: {loss:.5f}")

def main():
    data = np.array([
        [-2, -1],
        [25, 6],
        [17, 4],
        [-15, -6]
    ])
    all_y_trues = np.array([1, 0, 0, 1])
    network = OurNeuralNetwork()
    network.train(data, all_y_trues)

    # Predictions
    emily = np.array([-7, -3])
    frank = np.array([20, 2])
    print("Emily: %.3f" % network.feedforward(emily))
    print("Frank: %.3f" % network.feedforward(frank))

if __name__ == '__main__':
    main()
