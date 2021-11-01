import sys
sys.path += ['layers']
import numpy as np

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = False

# You can modify the imports of this section to indicate
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ['pyc_code']
    from inference_ import inference
    from calc_gradient_ import calc_gradient
    from loss_crossentropy_ import loss_crossentropy
    from update_weights_ import update_weights
else:
    # import your own implementation
    from inference import inference
    from calc_gradient import calc_gradient
    from loss_crossentropy import loss_crossentropy
    from update_weights import update_weights

######################################################

def train(model, input, label, params, numIters):
    '''
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [num_inputs]
        label: [num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
    '''
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", .02)
    # Weight decay
    wd = params.get("weight_decay", 0.0005)
    # Batch size
    batch_size = params.get("batch_size", 128)
    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", 'model.npz')

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr,
                     "weight_decay": wd,
                     "rho": .95,
                     "vx": []}

    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,))
    position = 0

    for i in range(numIters):
        # TODO: One training iteration
        # Steps:
        #   (1) Select a subset of the input to use as a batch
        batch = input[:,:,:,position:position+batch_size]
        batch_label = label[position:position+batch_size]
        position += batch_size
        position = position % num_inputs

        #   (2) Run inference on the batch
        values, activation = inference(model, batch)

        #   (3) Calculate loss and determine accuracy
        loss, output = loss_crossentropy(values, batch_label, params, True)

        #   (4) Calculate gradients
        grads = calc_gradient(model, batch, activation, output)

        #   (5) Update the weights of the model
        if len(update_params['vx']) == 0:
            update_params['vx'] = grads
        else:
            for j in range(len(grads)):
                update_params['vx'][j]['W'] = update_params['rho'] * update_params['vx'][j]['W'] + grads[j]['W']
                update_params['vx'][j]['b'] =  update_params['rho'] * update_params['vx'][j]['b'] + grads[j]['b']

        model = update_weights(model, grads, update_params)

        # Optionally,
        #   (1) Monitor the progress of training
        if i % 5 == 0:
            np.savez(save_file, **model)
            best = []

            val_values, val_activation = inference(model, params['val_data'])
            val_loss, output = loss_crossentropy(val_values, params['val_label'], {}, True)
            for k in range(val_values.shape[1]):
                best.append(np.where(val_values[:, k] == np.max(val_values[:, k]))[0][0])
            best = np.array(best)
            accuracy = np.sum(best == params['val_label']) / np.size(params['val_label'])

            print("Iteration: %d   Batch_loss: %.2f   Val_Accuracy: %.4f      Val_loss: %.4f" %
                  (i, loss, accuracy, val_loss))
            if accuracy > .98:
                break

    return model, loss

