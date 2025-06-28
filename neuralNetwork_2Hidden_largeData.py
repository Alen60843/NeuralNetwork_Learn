import numpy as np
import pandas as pd



def sigmoid(x):
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss (y_pred, y_true):
    return np.mean((y_true - y_pred) ** 2)

class OurNeuralNetwork:
    def __init__(self, input_size , first_hidden_size = 6, second_hidden_size = 6):
        self.input_size = input_size
        self.first_hidden_size = first_hidden_size
        self.second_hidden_size = second_hidden_size

        self.weights = {
            'h1': np.random.normal(size=(first_hidden_size, input_size)),# 6x2
            'h2': np.random.normal(size=(second_hidden_size, first_hidden_size)),# 3x6
            'o': np.random.normal(size=(second_hidden_size)) # Output layer weights
            }
        

        self.biases = {
            'h1': np.random.normal(size=(first_hidden_size,)),
            'h2': np.random.normal(size=(second_hidden_size,)),
            'o': np.random.normal()  # Output layer bias
        }

    def feedforward(self, x):
        sum_h1 = np.dot(self.weights['h1'], x) + self.biases['h1']
        h1 = sigmoid(sum_h1)
        sum_h2 = np.dot(self.weights['h2'], h1) + self.biases['h2']
        h2 = sigmoid(sum_h2)
        sum_o = np.dot(self.weights['o'], h2) + self.biases['o']
        o = sigmoid(sum_o)
        return o
    
    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 10000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # Forward pass
                sum_h1 = np.dot(self.weights['h1'], x) + self.biases['h1']
                h1 = sigmoid(sum_h1)
                sum_h2 = np.dot(self.weights['h2'], h1) + self.biases['h2']
                h2 = sigmoid(sum_h2)
                sum_o = np.dot(self.weights['o'], h2) + self.biases['o']
                o = sigmoid(sum_o)


                # Backpropagation
                d_L_d_ypred = -2 * (y_true - o) 
                d_ypred_d_sum_o = derivative_sigmoid(sum_o)

                # Output neuron gradients
                d_sum_o_d_w_o = h2

                d_sum_o_d_h2 = self.weights['o']
                d_L_d_sum_o = d_L_d_ypred * d_ypred_d_sum_o

                d_L_d_w_o = d_L_d_sum_o * d_sum_o_d_w_o
                d_L_d_b_o = d_L_d_sum_o 

                # 2nd hidden layer gradients
                d_h2_d_sum_h2 = derivative_sigmoid(sum_h2)
                d_L_d_h2 = d_L_d_sum_o * d_sum_o_d_h2
                d_L_d_sum_h2 = d_L_d_h2 * d_h2_d_sum_h2
                d_L_d_w_h2 = np.outer(d_L_d_sum_h2, h1)
                d_L_d_b_h2 = d_L_d_sum_h2

                # 1st hidden layer gradients
                d_sum_h2_d_h1 = self.weights['h2'].T  # shape (6,3)
                d_L_d_h1 = np.dot(d_sum_h2_d_h1, d_L_d_sum_h2)
                d_h1_d_sum_h1 = derivative_sigmoid(sum_h1)
                d_L_d_sum_h1 = d_L_d_h1 * d_h1_d_sum_h1
                d_L_d_w_h1 = np.outer(d_L_d_sum_h1, x)
                d_L_d_b_h1 = d_L_d_sum_h1

                # Update 
                self.weights['o'] -= learn_rate * d_L_d_w_o
                self.biases['o'] -= learn_rate * d_L_d_b_o

                self.weights['h2'] -= learn_rate * d_L_d_w_h2
                self.biases['h2'] -= learn_rate * d_L_d_b_h2

                self.weights['h1'] -= learn_rate * d_L_d_w_h1
                self.biases['h1'] -= learn_rate * d_L_d_b_h1
                
            if epoch % 200 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                # Uncomment to log:
                #print(f"Epoch {epoch}, Loss: {loss:.5f}")

               
def main():
    data_file = 'https://technion046195.netlify.app/datasets/wdbc.csv'
    dataset = pd.read_csv(data_file)

    dataset = dataset.drop(columns=dataset.columns[0])

    dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})

    x = dataset.iloc[:, 1:].values.astype(np.float32)  # Features
    y = dataset.iloc[:, 0].values.astype(np.float32)  # Labels
    x = (x - x.mean(axis=0)) / x.std(axis=0)  # Normalize features

    input_size = x.shape[1]
    network = OurNeuralNetwork(input_size)

    network.train(x, y)

    for i in range(10 ,40):
        pred = network.feedforward(x[i])
        print(f"Sample {i}: Predicted: {pred:.3f}, True: {y[i]}")




if __name__ == '__main__':
    main()
                




