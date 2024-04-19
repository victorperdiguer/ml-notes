# neuron and machine learning tests
# Coding a neuron. A neuron receives:
# - inputs from previous neurons
# - these inputs have a weight associated

#Coding an input neuron

import numpy as np

inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)

#Modelling multiple neurons - one set of inputs that go

inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1] #weights of neuron 1
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [
    # Neuron 1:
    inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
    # Neuron 2:
    inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
    # Neuron 3:
    inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3
]
print(outputs)

#------------------- NUMPY AND DOT PRODUCT ---------------------

import numpy as np

inputs = [1, 2, 3, 2.5]

# Combine weights into a matrix
weights = [
    [0.2, 0.8, -0.5, 1],   # Weights of neuron 1
    [0.5, -0.91, 0.26, -0.5], # Weights of neuron 2
    [-0.26, -0.27, 0.17, 0.87] # Weights of neuron 3
]

# Combine biases into a vector
biases = [2, 3, 0.5]

# Calculate output using dot-product and adding biases
output = np.dot(weights, inputs) + biases
print(output)

#------------------- WORKING WITH BATCHES -----------------------

#Working with batches of inputs (list of inputs)

print("Batches example:")
inputBatch = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3, -0.8]]
weights = [
    [0.2, 0.8, -0.5, 1],   # Weights of neuron 1
    [0.5, -0.91, 0.26, -0.5], # Weights of neuron 2
    [-0.26, -0.27, 0.17, 0.87] # Weights of neuron 3
]
#we have to convert the list into a numpy array because we want to apply the Transposition to the weight array
output = np.dot(inputBatch, np.array(weights).T) + biases 
print(output)