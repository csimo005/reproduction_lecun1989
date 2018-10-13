import numpy as np

def sigmoid_forward(input_values): #passes input through sigmoid activation
    output = 1/(1+np.exp(-1*input_values)) # 1/(1+e^-x)\
    return output

def sigmoid_derivative(input_values, output_values): #calculate derivate for each inpur output pair
    derivative = output_values*(1+output_values)
    return derivative

activation_functions = {
    'sigmoid':sigmoid_forward
}

activation_derivatives = {
    'sigmoid':sigmoid_derivative
}

