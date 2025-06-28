import torch # base library
import torch.nn as nn # tools to define layers, activation, loss functions, etc.
import torch.optim as optim # build in optimizers like SGD, Adam, etc.
from sklearn.model_selection import train_test_split # split data into train and test sets
from sklearn.preprocessing import StandardScaler # standardize features by removing the mean and scaling to unit variance
import pandas as pd 

class NerualNetwork(nn.Module):
    def __init__(self, input_size, hidden1 = 30, hidden2 = 20, hidden3 = 10):
        super(NerualNetwork, self).__init__()

        # Define the layers
        '''
        nn.Linear(in_features, out_features)
        creates a weight matrix of shape (out_features, in_features) and a
        bias vector of shape (out_features,).
        '''
        self.fc1 = nn.Linear(input_size, hidden1)  # First hidden layer
        self.fc2 = nn.Linear(hidden1, hidden2)      # Second hidden layer
        self.fc3 = nn.Linear(hidden2, hidden3)      # Third hidden layer
        self.output = nn.Linear(hidden3, 1)         # Output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x)) # Apply sigmoid activation to the first layer
        x = torch.sigmoid(self.fc2(x)) # Apply sigmoid activation to the second layer
        x = torch.sigmoid(self.fc3(x)) # Apply sigmoid activation to the third layer
        x = torch.sigmoid(self.output(x)) # Apply sigmoid activation to the output layer
        return x
    
# Training the model
def main():
    data_file = 'https://technion046195.netlify.app/datasets/wdbc.csv'
    df = pd.read_csv(data_file)
    df = df.drop(columns=df.columns[0])  # Drop the first column (ID)
    df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0}) # Map 'M' to 1 and 'B' to 0
    
    x = df.iloc[:, 1:].values
    y = df.iloc[:,0].values

    # Normalize the features
    scalar = StandardScaler()
    x = scalar.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)  # Reshape y to be a column vector
    x_test = torch.tensor(x_test, dtype=torch.float32) 
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    model = NerualNetwork(x.shape[1])  # Initialize the model with input size
    criterion = nn.BCELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=0.1)  # Stochastic Gradient Descent optimizer  

    epochs = 20000
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x_train)
        # Calculate the loss
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()  # clear the gradients
        loss.backward()  # backpropagation
        optimizer.step()  # update the weights

        if epoch % 1000 == 0:  
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


    with torch.no_grad():
        preds = model(x_test)
        predicted_labels = (preds > 0.5).float()
        accuracy = (predicted_labels == y_test).float().mean()
        print(f'Accuracy: {accuracy.item() * 100:.3f}%')

                  

if __name__ == "__main__":
    main()    

    '''
   * float32 means a 32-bit floating-point number.
    It can represent decimal numbers with ~7 decimal digits of precision.
    Use .float32 whenever you're working with neural network inputs, weights,
    and losses in PyTorch 

   * .float() is a PyTorch tensor method that converts a tensor to float32.

   * when we compare model(x_test) > 0.5 we get bollean tensor but 
    we need numbers: (True == True) -> 1, (False == True) -> 0
    so .float() converts True to 1.0 and False to 0.0
     
    * random_state=42 sets the seed for the random number generator.
        That way:
            Every time you run the code, you get the same random split.
            Useful for reproducibility.

    * fit_transform: first calculates the mean and standard deviation of each feature.
                     Then it applies standardization: x_scaled = (x - mean) / std
        Why it's important:
            Neural networks train much faster and more reliably when input values are centered around 0 with similar scales.
            If one feature is in the range [0, 1] and another is [1,000,000, 2,000,000], gradients become unstable and learning becomes inefficient.
        ---Always normalize or standardize your input data before training neural networks-----  
         
    * BCELoss: loss = -[y * log(y_pred) + (1 - y) * log(1 - y_pred)]
    - Use BCELoss when your model outputs probabilities (between 0 and 1) and you want to measure how well it predicts binary labels (0 or 1).
    - Final layer is sigmoid activation, which outputs values between 0 and 1.     
    '''