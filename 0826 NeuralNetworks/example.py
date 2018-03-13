import math
import random
import numpy as np

def sigmoid(t):
    return 1/(1+math.exp(-t))

def neuron_output(w, inp, bias=0):
    return sigmoid(bias+ np.dot(w, inp))

def feed_forward(neural_network, inp):
 # inp: vector, neural_network: theta list for each layers
    outputs=[]
    for layer in neural_network:
        inp_bias= inp + [1]
        output= [neuron_output(neuron, inp_bias) for neuron in layer]
        outputs.append(output)
        inp = output
    return outputs

def step_function(x):
    return 1 if x>=0 else 0

def percepron_output(w, x, bias=0):
    return step_function(np.dot(w, x) +bias)

def backpropagate(network, inp, targets):
    hidden_outputs, outputs= feed_forward(network, inp)
    #1st
    output_deltas= [output*(1-output)*(output-target) \
                    for output, target in zip(outputs, targets)]
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i]* hidden_output
    #2nd
    hidden_deltas= [hidden_output*(1-hidden_output)* \
                    (np.dot(output_deltas, [n[i] for n in output_layer]))\
                    for i, hidden_output in enumerate(hidden_outputs)]
    for i, hidden_neuron in enumerate(network[0]):
        for j, inp_val in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i]* inp_val
    pass

def predict(inp):
    return feed_forward(network, inp)[-1]

# print(sigmoid(0))
targets= [[1 if i==j else 0 for i in range(10)] for j in range(10)]
input_size= 25
num_hidden= 5
output_size= 10
hidden_layer= [[random.random() for _ in range(input_size+ 1)]\
                for _ in range(num_hidden)]
output_layer= [[random.random() for _ in range(num_hidden+1)]\
                for _ in range(output_size)]
network= [hidden_layer, output_layer]

inputs = [[1,1,1,1,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,1,1,1,1],
         [0,0,1,0,0,
         0,0,1,0,0,
         0,0,1,0,0,
         0,0,1,0,0,
         0,0,1,0,0],
         [1,1,1,1,1,
         0,0,0,0,1,
         1,1,1,1,1,
         1,0,0,0,0,
         1,1,1,1,1],
         [1,1,1,1,1,
         0,0,0,0,1,
         1,1,1,1,1,
         0,0,0,0,1,
         1,1,1,1,1],
         [1,0,0,0,1,
         1,0,0,0,1,
         1,1,1,1,1,
         0,0,0,0,1,
         0,0,0,0,1],
         [1,1,1,1,1,
         1,0,0,0,0,
         1,1,1,1,1,
         0,0,0,0,1,
         1,1,1,1,1],
         [1,1,1,1,1,
         1,0,0,0,0,
         1,1,1,1,1,
         1,0,0,0,1,
         1,1,1,1,1],
         [1,1,1,1,1,
         0,0,0,0,1,
         0,0,0,0,1,
         0,0,0,0,1,
         0,0,0,0,1],
         [1,1,1,1,1,
         1,0,0,0,1,
         1,1,1,1,1,
         1,0,0,0,1,
         1,1,1,1,1],
         [1,1,1,1,1,
         1,0,0,0,1,
         1,1,1,1,1,
         0,0,0,0,1,
         1,1,1,1,1]]

for _ in range(10000):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)

print(predict(inputs[7]))
print(predict([0,1,1,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,1,1,0,1,1,1,0]))
