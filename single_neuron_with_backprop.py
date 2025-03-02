import numpy as np 

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred)**2)

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float,
                 epochs: int) -> tuple[np.ndarray, float, list[float]]:
    weights = initial_weights
    bias = initial_bias 
    mse_losses = []
    
    for epoch in range(epochs):
        x = np.dot(features, weights) + bias 
        y_pred = sigmoid(x)
        loss = mse_loss(labels, y_pred)
        mse_losses.append(loss)
        
        # dloss/dw = dloss/dypred * dy_pred/dw
        # furthur breakdown of dy_pred/dw
        # dy_pred/dw = dy_pred/dx * dx/dw
        
        # dloss/db = dloss/dypred * dy_pred/db
        # furthur breakdown of dy_pred/db
        # dy_pred/db = dy_pred/dx * dx/db
        # dx/db = 1
        
        dloss_dypred = 2 * (y_pred - labels) / len(labels)
        dy_pred_dx = sigmoid_derivative(x)
        dy_pred_db = sigmoid_derivative(x) * 1
        
        dloss_dw = np.dot(features.T, dloss_dypred * dy_pred_dx)
        dloss_db = np.sum(dloss_dypred * dy_pred_db)
        
        weights -= learning_rate * dloss_dw
        bias  -= learning_rate * dloss_db
        
    return weights, bias, mse_losses

def main():
    result = train_neuron(np.array([[1, 2], [2, 3], [3, 1]]), np.array([1, 0, 1]), np.array([0.5, -0.2]), 0, 0.1, 3)
    print(result)

if __name__ == "__main__":
    main()





