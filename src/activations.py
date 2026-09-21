import numpy as np

# Activation function sigmoid
def sigmoid(x: np.ndarray) -> np.ndarray:
    
    # input verification
    assert isinstance(x, np.ndarray), "error - input must be numpy array"
    
    # function
    result = 1 / (1 + np.exp(-x))
    
    # output verification
    assert np.all((result >= 0) & (result <= 1)), "error - result should be between 0 and 1"
                  
    return result

# derivative of sigmoid
def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    
    # input verification
    assert isinstance(x, np.ndarray), "error - input must be numpy array"
    
    # function
    result = sigmoid(x) * (1 - sigmoid(x))
    
    # output verification
    assert np.all(result >= 0), "error - result should be greater or equal to 0"

    return result

# activation function ReLU 
def relu(x: np.ndarray) -> np.ndarray:

    """
    ReLU activation: max (0, x)
    """

    assert isinstance(x, np.ndarray), "error - input must be numpy array"
    result =  np.maximum(0, x)
    assert np.all(result >= 0), "error - output must be non-negative"

    return result

# derivative of ReLU
def relu_derivative(x: np.ndarray) -> np.ndarray:

    """
    Derivative of ReLU: 1 if x > 0, else 0
    """

    assert isinstance(x, np.ndarray), "error - input must be numpy array"
    result = np.where(x > 0, 1, 0)
    assert np.all((result == 0) | (result == 1)), "error - result should be 0 or 1"

    return result


# activation function Softmax
def softmax(x: np.ndarray) -> np.ndarray:

    """
    Softmax activation: exp(x) / sum (exp(x))
    """
    
    assert isinstance(x, np.ndarray), "error - input must be numpy array"
    exp_x = np.exp(x - np.max(x , axis=-1, keepdims=True))
    result = exp_x / exp_x.sum(axis=-1, keepdims=True)
    assert np.all((result >= 0) & (result <= 1)), "error - Softmax output must be in [0, 1]"
    assert np.allclose(np.sum (result, axis=1), 1), "error - Softmax output must sum to 1 per sample"
    return result


# activation function Leaky relu
def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:

    assert isinstance(x, np.ndarray), "error - input must be numpy array"
    assert alpha > 0, "error - alpha must be > 0"
    return np.where(x > 0 , x, alpha * x)

# derivative of Leaky relu
def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:

    assert isinstance(x, np.ndarray), "error - input must be numpy array"
    assert alpha > 0, "error - alpha must be > 0"

    result = np.where(x > 0, 1, alpha)

    assert np.all((result == 1) | (result == alpha)), "error - result should be 1 or alpha"

    return result