import numpy as np


def update_weights(model, grads, hyper_params):
    '''
    Update the weights of each layer in your model based on the calculated gradients
    Args:
        model: Dictionary holding the model
        grads: A list of gradients of each layer in model["layers"]
        hyper_params: 
            hyper_params['learning_rate']
            hyper_params['weight_decay']: Should be applied to W only.
    Returns: 
        updated_model:  Dictionary holding the updated model
    '''
    num_layers = len(grads)
    a = hyper_params["learning_rate"]
    lmd = hyper_params["weight_decay"]
    rho = hyper_params['rho']
    vx = hyper_params['vx']
    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients

    for i in range(num_layers):
        update = -1 * a * vx[i]['W']
        updated_model['layers'][i]['params']['W'] += update - updated_model['layers'][i]['params']['W'] * lmd
        update = -1 * a *  vx[i]['b']
        updated_model['layers'][i]['params']['b'] += update

    return updated_model
