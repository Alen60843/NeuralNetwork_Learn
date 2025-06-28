import torch # base library
import torch.nn as nn # tools to define layers, activation, loss functions, etc.
import torch.optim as optim # build in optimizers like SGD, Adam, etc.
from sklearn.model_selection import train_test_split # split data into train and test sets
from sklearn.preprocessing import StandardScaler # standardize features by removing the mean and scaling to unit variance
from sklearn.feature_extraction.text import CountVectorizer # convert a collection of text documents to a matrix of token counts
import pandas as pd 
import argparse

class NerualNetwork(nn.Module):
    def __init__(self, input_size, hidden1 = 40, hidden2 = 35, hidden3 = 30, hidden4 = 20):
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
        self.fc4 = nn.Linear(hidden3, hidden4)      # Fourth hidden layer
        self.output = nn.Linear(hidden4, 1)         # Output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x)) # Apply sigmoid activation to the first layer
        x = torch.sigmoid(self.fc2(x)) # Apply sigmoid activation to the second layer
        x = torch.sigmoid(self.fc3(x)) # Apply sigmoid activation to the third layer
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.output(x)) # Apply sigmoid activation to the output layer
        return x
    
def load_and_prepare_data(data_file):
    df = pd.read_csv(data_file)
    
    df['Contagious'] = df['Contagious'].map({True: 1.0, False: 0.0})  # Convert boolean to float
    df['Chronic'] = df['Chronic'].map({True: 1.0, False: 0.0})
    df['combined_text'] = (
                            df['Symptoms'].fillna('') + ' ' + 
                            df['Treatments'].fillna('') + ' ' + 
                            df['Contagious'].astype(str) )


    vectorizer = CountVectorizer(max_features=1000)  # Limit to 1000 features
    x = vectorizer.fit_transform(df['combined_text'].fillna("")).toarray()  # Convert symptoms to a matrix of token counts
    y = df['Chronic'].values  

    # Normalize the features
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    return x, y

def train_the_model(x, y, input_size, model_path = 'model.pth'):
    # Split into 80% train and 20% test
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
    # split to train set into 75% train and 25% validation
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size = 0.25, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)  # Reshape y to be a column vector
    
    x_val = torch.tensor(x_val, dtype = torch.float32)
    y_val = torch.tensor(y_val.reshape(-1,1), dtype = torch.float32)

    x_test = torch.tensor(x_test, dtype = torch.float32)
    y_test = torch.tensor(y_test.reshape(-1,1), dtype = torch.float32)


    model = NerualNetwork(x.shape[1])  # Initialize the model with input size
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss 
    optimizer = optim.Adam(model.parameters(), lr=0.1)  # Stochastic Gradient Descent optimizer  

    epochs = 10000
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x_train)
        # Calculate the loss
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()  # clear the gradients
        loss.backward()  # backpropagation
        optimizer.step()  # update the weights

        # Evaluate on validation set
        with torch.no_grad():
            val_preds = model(x_val)
            val_loss = criterion(val_preds, y_val)
            val_accuracy = (val_preds > 0.5).float().eq(y_val).float().mean()
        if epoch % 1000 == 0:  
            print(f'Epoch {epoch}, Loss: {loss.item():.7f} \n Training Loss: {loss.item():.7f}, Validation Loss: {val_loss.item():.7f} \n Validation Accuracy: {val_accuracy.item() * 100:.6f}%')
    
    torch.save(model.state_dict(), model_path)  # Save the model state
    print(f'Model saved to {model_path}')

def test_the_model(x, y ,input_size, model_path = 'model.pth'):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
    x_test = torch.tensor(x_test, dtype=torch.float32) 
    y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    model = NerualNetwork(input_size)  # Initialize the model with input size
    model.load_state_dict(torch.load(model_path))  # Load the saved model state
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        print("Testing the model...")
        preds = model(x_test)
        predicted_labels = (preds > 0.5).float()
        accuracy = (predicted_labels == y_test).float().mean()
        print(f'Accuracy: {accuracy.item() * 100:.3f}%')


def main():

    parser = argparse.ArgumentParser(description='Train or test a neural network model for disease prediction.')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--data', type=str, default='Data_sets/Diseases_Symptoms.csv', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save/load model')
    args = parser.parse_args()

    x, y = load_and_prepare_data(args.data)
    input_size = x.shape[1]

    if args.train:
        print("Training the model...")
        train_the_model(x, y, input_size, args.model_path)
    if args.test:
        print("Testing the model...")
        test_the_model(x, y, input_size, args.model_path)
    if not args.train and not args.test:
        print("Please specify --train or --test to run the model.")  
   
if __name__ == "__main__":
    main()    

    