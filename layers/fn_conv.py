import numpy as np
import scipy.signal


def fn_conv(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [filter_height] x [filter_width] x [filter_depth] x [num_filters] array
            params['b']: layer bias, [num_filters] x 1 array
        hyper_params: Optional, could include information such as stride and padding.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_filters] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """

    in_height, in_width, num_channels, batch_size = input.shape
    _, _, filter_depth, num_filters = params['W'].shape
    out_height = in_height - params['W'].shape[0] + 1
    out_width = in_width - params['W'].shape[1] + 1

    assert params['W'].shape[2] == input.shape[2], 'Filter depth does not match number of input channels'

    # Initialize
    output = np.zeros((out_height, out_width, num_filters, batch_size))
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}
    
    # TODO: FORWARD CODE
    #       Update output with values
    for k in range(batch_size):
        for i in range(num_filters):
            image = scipy.signal.convolve(input[:, :, :, k], np.flip(params['W'][:, :, :, i]), mode='valid')
            image = np.reshape(image, (out_height, out_width)) + params['b'][i]
            output[:, :, i, k] = image




    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)
        
        # TODO: BACKPROP CODE
        #       Update dv_input and grad with values
        # Calculate grad['W']
        for k in range(batch_size):
            for i in range(filter_depth):
                for j in range(num_filters):
                    grad['W'][:, :, i, j] = grad['W'][:, :, i, j] + \
                                            scipy.signal.convolve(input[:, :, i, k], np.flip(dv_output[:, :, j, k]), mode='valid')
        grad['W'][:, :, :, :] = grad['W'][:, :, :, :] / batch_size

        # Calculate grad['b']
        b_deriv = np.ones(output.shape)
        out = []
        for j in range(num_filters):
            total = 0
            for i in range(num_filters):
                avg = 0
                for k in range(batch_size):
                    avg = avg + scipy.signal.convolve(b_deriv[:, :, i, k], np.flip(dv_output[:, :, j, k]), mode='valid')
                total = total + avg/batch_size
            out.append(total/num_filters)
        out = np.array(out)
        grad['b'] = np.reshape(out, grad['b'].shape)

        # Compute dv_input
        for k in range(batch_size):
            for i in range(num_channels):
                avg = np.zeros((in_height, in_width))
                for j in range(num_filters):
                    avg = avg + scipy.signal.convolve(params['W'][:, :, i, j], dv_output[:, :, j, k], mode='full')
                dv_input[:, :, i, k] = avg

    return output, dv_input, grad
