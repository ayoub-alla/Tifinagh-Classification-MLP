import numpy as np
 
# MSE loss function
def MSE(y_true: np.ndarray , y_pred: np.ndarray  ) -> np.float64:
    return np.mean(np.power( y_true - y_pred  , 2))

# Categorical Cross-Entropy loss function
def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """
    Categorical Cross-Entropy: J = -1/m sum(y_true log(y_pred))
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "error - Inputs to loss must be numpy arrays"
    assert y_true.shape == y_pred.shape, "error - y_true and y_pred must have the same shape"

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    assert not np.isnan(loss), "error - Loss computation resulted in NaN"

    return loss