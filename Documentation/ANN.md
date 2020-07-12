## ARTIFICIAL NEURAL NETWORK

- Define independent variables and dependent variable
- Define Hyperparameters
- Define Activation Function and its derivative
- Train the model
- Make predictions

Class MLP implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.

**MLP class has following functions:**

## init(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2)
  *Constructor for the MLP. Takes the number of inputs,
        a variable number of hidden layers, and number of outputs
        Args:
        num_inputs (int): Number of inputs
        hidden_layers (list): A list of ints for the hidden layers
        num_outputs (int): Number of outputs*
## forward_propagate(self, inputs):
  *Computes forward propagation of the network based on input signals.
       Args:
       inputs (ndarray): Input signals
       Returns:
       activations (ndarray): Output values*
## back_propagate(self, error):
 *Backpropogates an error signal.
    Args:
    error (ndarray): The error to backprop.
    Returns:
    error (ndarray): The final error of the input*
## train(self, inputs, targets, epochs, learning_rate):
  *Trains model running forward prop and backprop
    Args:
    inputs (ndarray): X
    targets (ndarray): Y
    epochs (int): Num. epochs we want to train the network for
    learning_rate (float): Step to apply to gradient descent*
## gradient_descent(self, learningRate=1):
  *Learns by descending the gradient
    Args:
    learningRate (float): How fast to learn.*

* Activation Function and it’s derivative: Our activation function is the sigmoid function.*
## _sigmoid(self, x):
  *Sigmoid activation function
    Args:
    x (float): Value to be processed
    Returns:
    y (float): Output*
## sigmoidderivative(self, x):
  *Sigmoid derivative function
    Args:
    x (float): Value to be processed
    Returns:
    y (float): Output*
## _mse(self, target, output):
  *Mean Squared Error loss function
    Args:
    target (ndarray): The ground trut
    output (ndarray): The predicted values
    Returns:
        (float): Output*
### Example Code
```
    from ann import MLP
    # create a dataset to train a network for the sum operation
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(items, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
```

<br />
<br />
<br />
<br />
