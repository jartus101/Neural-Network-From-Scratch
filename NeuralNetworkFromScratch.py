import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#chosen activation function for now
def reLU(x):
    return np.maximum(0, x)

#(d/dx)(x)=1, (d/dx)(0)=0
def reLU_derivative(x):
    return np.where(x > 0, 1, 0)

#make a matrix based on the intended 
def one_hot(y):
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Labels must be integer values.")
    num_classes = np.max(y) + 1  # Ensure num_classes is an integer
    one_hot_y = np.zeros((y.size, num_classes))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y.T

#found online
def soft_max(x):
    shift_x = x - np.max(x, axis=0, keepdims=True)
    exps = np.exp(shift_x)
    return exps / np.sum(exps, axis=0, keepdims=True)

#define everything such that the matrix multiplication works
def init_params():
    w1 = np.random.randn(10, 784) * 0.1
    b1 = np.random.randn(10, 1) * 0.1
    w2 = np.random.randn(10, 10) * 0.1
    b2 = np.random.randn(10, 1) * 0.1
    return w1, b1, w2, b2

#z represents the layer pre-activation, and a represents the layer post-activation
def forward_propagation(w1, b1, w2, b2, x):
    Z1 = np.dot(w1, x) + b1
    A1 = reLU(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = soft_max(Z2)
    return Z1, A1, Z2, A2

#calculate the gradient by finding the partial derivatives
def back_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = dZ2.dot(A1.T)/m
    db2 = np.sum(dZ2)/m
    dZ = W2.T.dot(dZ2)*reLU_derivative(Z1)
    dW = dZ.dot(X.T)/m
    db = np.sum(dZ)/m
    return dW, db, dW2, db2

#update the values by using the gradient relative to the learning rate
def update_vals(weights_one, biases_one, weights_two, biases_two, dW1, db1, dW2, db2, learning_rate):
    weights_one -= learning_rate * dW1
    biases_one -= learning_rate * db1
    weights_two -= learning_rate * dW2
    biases_two -= learning_rate * db2
    return weights_one, biases_one, weights_two, biases_two

#just compare the predictions against the actual output
def accuracy(predictions, Y):
    return np.mean(predictions == np.argmax(Y, axis=0)) * 100

def prediction(A2):
    return np.argmax(A2, 0)

def gradient_descent(X, Y, alpha, iterations, batch_size):
    W1, b1, W2, b2 = init_params()
    
    for i in range(iterations):
        #shuffles the whole thing
        permutation = np.random.permutation(X.shape[1])
        
        #takes the broken down values from it
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        for j in range(0, X.shape[1], batch_size):
            #implements the batch sizing so that it processes only 32 elements at a time
            end = j + batch_size
            X_batch = X_shuffled[:, j:end]
            Y_batch = Y_shuffled[:, j:end]

            #gradient descent algorithm: do forward propagation, then back propagation, then update values
            Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W1, W2, X_batch, Y_batch)
            W1, b1, W2, b2 = update_vals(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        #print the accuracy every 10 iterations
        if i % 10 == 0 or i == iterations - 1:
            Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
            predictions = prediction(A2)
            x_array.append(i)
            y_array.append(accuracy(predictions, Y))
            print(f"Iteration: {i}, Accuracy: {accuracy(predictions, Y):.2f}%")
    
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    X_reshaped = X.reshape(-1, 1)
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X_reshaped)
    predictions = prediction(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = x_test_reshaped[:, index]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = y_test[index]

    print("Prediction: ", prediction)
    print("Label: ", label)

    plt.imshow(x_test[index], cmap='gray')
    plt.show()
    
    return prediction==label

x_array = []
y_array = []

x_train_reshaped = x_train.reshape(60000, 784).T / 255.0
y_train_encoded = one_hot(y_train)

x_test_reshaped = x_test.reshape(x_test.shape[0], -1).T / 255.0
y_test_encoded = one_hot(y_test)

W1, b1, W2, b2 = gradient_descent(x_train_reshaped, y_train_encoded, alpha=0.1, iterations=80, batch_size=32)

for i in range(20):
    test_prediction(random.randint(1,9999), W1, b1, W2, b2)