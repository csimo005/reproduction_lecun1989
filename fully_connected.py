import numpy as np
from numpy.random import rand

from activation import activation_functions, activation_derivatives

"""
fc_layer is my implementation of a fully connected layer

some conventions:
input and outputs are expected to be a 2D column vectors e.g. vec.shape() = (m,1)
every element operation (i.e. activation functions)  will perserve this

"""

class fc_layer:
    def __init__(self, name, input_nodes, output_nodes, activation_func='sigmoid'):
        #make sure an integer number of nodes is passed in
        if not isinstance(input_nodes, int):
            raise ValueError('Expected an integer number of input nodes')
        if not isinstance(output_nodes, int):
            raise ValueError('Expected an integer number of output nodes')

        #make sure that a string that is a valid activation func key is passed in
        if not isinstance(activation_func, str):
            raise ValueError('Expected string value specifying activation function')
        
        global activation_functions
        if not activation_func in activation_functions:
            raise ValueError('Unknown activation funciton \'' + activation_func + '\' specified')
      
        global activation_derivatives
        if not activation_func in activation_derivatives:
            raise ValueError('Unknown activation funciton \'' + activation_func + '\' specified')

        #name should be any string, really only for debug purposes
        if not isinstance(name, str):
            raise ValueError('Layer name must be a string')

        self._name=name
        self._input_nodes=input_nodes
        self._output_nodes=output_nodes
        self._activation_func=activation_func
        self._weights = rand(self._output_nodes, self._input_nodes+1)

    def forward_pass(self, input_values):
        weighted_inputs = np.matmul(self._weights, np.concatenate((input_values,[[1]])))
        output_value = activation_functions[self._activation_func](weighted_inputs)
        return output_value

    def layer_derivative(self, input_values, output_values, previous_derivative):
        weighted_inputs = np.matmul(self._weights, np.concatenate((input_values,[[1]])))
        dy_dx = activation_derivatives[self._activation_func](weighted_inputs, output_values)
        dE_dx = previous_derivative*dy_dx

        dE_dw = np.matmul(dE_dx, np.transpose(np.concatenate((input_values,[[1]]))))
        dE_dy = np.matmul(np.transpose(self._weights), dE_dx)

        return dE_dw, dE_dy

    def update_weights(self, delta_w):
        self._weights = self._weights+delta_w
     
        return None

    def randomize_weights(self):
        self._weights = rand(self._output_nodes, self._input_nodes)

if __name__=='__main':
    layer = fc_layer('test',30,10,'sigmoid')
    output = layer.forward_pass(rand(30,1))
    if output is None:
        print 'forward_pass did not work'
    dE_dw, dE_dy = layer.layer_derivative(rand(30,1),rand(10,1),rand(10,1))
    if dE_dw is None or dE_dy is None:
        print 'layer_derivative did not work'
    layer.update_weights(dE_dw,0.1)
    
