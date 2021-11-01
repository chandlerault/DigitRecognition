import numpy as np

def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model['layers'])
    activations = [None,] * num_layers

    # TODO: FORWARD PROPAGATION CODE
    for i in range(num_layers):
        layer = model['layers'][i]
        function, params, hyper_params = layer['fwd_fn'], layer['params'], layer['hyper_params']
        activations[i], dv_input, grad = function(input, params, hyper_params, False)
        input = activations[i]
    output = activations[-1]
    return output, activations
