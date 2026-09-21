import numpy as np
from .activations import relu , relu_derivative , softmax
from .loss import categorical_cross_entropy



# Classe MultiClass Neural Network
class MultiClass_NN:

    """
    Initialize the neural network with given layer sizes and learning rate.
    layer_sizes: List of integers [input_size, hidden_size, output_size]
    """
    def __init__(self, layer_sizes, learning_rate = 0.01 , lambda_reg=0.01):

        assert isinstance (layer_sizes, list) and len(layer_sizes) >= 2, "layer_sizes must be a list with at least 2 elements"
        assert all(isinstance (size, int) and size > 0 for size in layer_sizes), "All layer sizes must be positive integers"
        assert isinstance (learning_rate, (int, float)) and learning_rate > 0, "Learning rate must be a positive number"

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        self.lambda_reg = lambda_reg

        # Initialisation des poids et biais
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes [i], layer_sizes [i+1]) * 0.01
            b = np.zeros((1, layer_sizes [i+1]))
            assert w.shape == (layer_sizes [i], layer_sizes [i+1]), f"Weight matrix {i+1} has incorrect shape"
            assert b.shape == (1, layer_sizes [i+1]), f"Bias vector {i+1} has incorrect shape"
            self.weights.append(w)
            self.biases.append(b)

    def forward (self, X):
        """
        Forward propagation: Z^{[l]}=A^{[l-1]} W^{[l]}+b^{[l]} , A^{[l]} = g(Z^{[l]})
        """
        assert isinstance (X, np.ndarray), "Input X must be a numpy array"
        assert X.shape [1] == self.layer_sizes [0], f"Input dimension ({X.shape [1]}) must match input layer size ({self.layer_sizes [0]})"
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights) - 1):
            z = self.activations[i] @ self.weights[i] + self.biases[i]
            assert z.shape == (X.shape[0], self.layer_sizes [i+1]), f"Z^{[i+1]} has incorrect shape"
            self.z_values.append(z)
            self.activations.append (relu(z))

        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        assert z.shape == (X.shape[0], self.layer_sizes [-1]), "Output Z has incorrect shape"
        self.z_values.append(z)

        output = softmax(z)
        assert output.shape == (X.shape[0], self.layer_sizes [-1]), "Output A has incorrect shape"

        self.activations.append (output)
        
        return self.activations [-1]

    def compute_loss(self, y_true, y_pred):
      return categorical_cross_entropy(y_true=y_true , y_pred=y_pred)

    def compute_accuracy(self, y_true, y_pred):

        """
        Compute accuracy: proportion of correct predictions
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to accuracy must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(predictions == true_labels)
        
        assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"
        return accuracy

    def backward (self, X, y, outputs):
        """
        Backpropagation
        compute dW^{[l]}, db^{[l]} for each layer
        """
        assert isinstance (X, np.ndarray) and isinstance (y, np.ndarray) and isinstance (outputs, np.ndarray), "Inputs to backward must be numpy arrays"
        assert X.shape[1] == self.layer_sizes [0], f"Input dimension ({X.shape [1]}) must match input layer size ({self.layer_sizes [0]})"
        assert y.shape == outputs.shape, "y and outputs must have the same shape"

        m = X.shape [0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        dZ = outputs - y # Gradient pour softmax cross-entropy
        assert dZ.shape == outputs.shape, "dZ for output layer has incorrect shape"
        self.d_weights [-1] = (self.activations [-2].T @ dZ) / m
        # L2 lambda
        self.d_weights[-1] += self.lambda_reg * self.weights[-1] / m
        self.d_biases [-1] = np.sum(dZ, axis = 0, keepdims=True) / m

        for i in range(len(self.weights) - 2, -1, -1):
            dZ = (dZ @ self.weights[i+1].T) * relu_derivative(self.z_values[i])
            assert dZ.shape == (X.shape[0], self.layer_sizes [i+1]), f"dZ^{[i+1]} has incorrect shape"
            self.d_weights [i] = (self.activations[i].T @ dZ) / m
            self.d_biases [i] = np.sum(dZ, axis=0, keepdims=True) / m

            #dW^{[l]} += lambda * W^{[l]} / m, ou lambda est le coefficient de regularisation
            self.d_weights[i] += self.lambda_reg * self.weights[i] / m

        for i in range(len(self.weights)):
            self.weights [i] -= self.learning_rate * self.d_weights [i]
            self.biases [i] -= self.learning_rate * self.d_biases [i]

    def train (self, X, y, X_val, y_val, epochs, batch_size , lr_e:int=0 , lr_dv:float=0.5):
        """
        Train the neural network using mini-batch SGD, with validation
        """
        assert isinstance (X, np.ndarray) and isinstance (y, np.ndarray), "X and y must be numpy arrays"
        assert isinstance (X_val, np.ndarray) and isinstance(y_val, np.ndarray), "X_val and y_val must be numpy arrays"
        assert X.shape[1] == self.layer_sizes [0], f"Input dimension ({X.shape [1]}) must match input layer size ({self.layer_sizes [0]})"
        assert y.shape[1] == self.layer_sizes [-1], f"Output dimension ({y.shape[1]}) must match output layer size ({self.layer_sizes [-1]})"
        assert X_val.shape [1] == self.layer_sizes [0], f"Validation input dimension ({X_val.shape [1]}) must match input layer size ({self.layer_sizes [0]})"
        assert y_val.shape[1] == self.layer_sizes [-1], f"Validation output dimension ({y_val.shape[1]}) must match output layer size ({self.layer_sizes [-1]})"
        assert isinstance (epochs, int) and epochs > 0, "Epochs must be a positive integer"
        assert isinstance (batch_size, int) and batch_size > 0, "Batch size must be a positive integer"

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range (epochs):
            indices = np.random.permutation (X.shape [0])
            X_shuffled = X [indices]
            y_shuffled = y [indices]
            epoch_loss = 0

            if lr_e != 0 and epoch != 0:
                if epoch % lr_e == 0:
                    self.learning_rate = self.learning_rate * lr_dv


            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled [i:i+batch_size]
                y_batch = y_shuffled [i:i+batch_size]
                outputs = self.forward (X_batch)
                epoch_loss += self.compute_loss(y_batch, outputs)
                self.backward (X_batch, y_batch, outputs)

            #Calculer les pertes et accuracies
            train_loss = epoch_loss / (X.shape [0] // batch_size)
            train_pred = self.forward (X)
            train_accuracy = self.compute_accuracy (y, train_pred)

            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_accuracy = self.compute_accuracy (y_val, val_pred)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

            if epoch % 10 == 0:
                print (f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                       f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f} , lr = {self.learning_rate}")

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        """
        Predict class labels
        """
        assert isinstance (X, np.ndarray), "Input X must be a numpy array"
        assert X.shape [1] == self.layer_sizes [0], f"Input dimension ({X.shape [1]}) must match input layer size ({self.layer_sizes [0]})"
        outputs = self.forward (X)
        predictions = np.argmax(outputs, axis=1)
        assert predictions.shape == (X.shape[0],), "Predictions have incorrect shape"
        return predictions
