from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Lambda, add
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.backend import clear_session

def build_model(network):
    clear_session()

    def MaxAE(y_true, y_pred):
        minus_y_pred = Lambda(lambda x: -x)(y_pred)
        return K.max(K.abs(add([y_true, minus_y_pred])))

    n_layers = network['network_config']['n_layers']
    n_nodes = network['layer_config']['n_nodes']
    layer_types = network['layer_config']['layer_type']
    input_shape = network['io_config']['input_shape']
    output_shape = network['io_config']['output_shape']
    activations = network['layer_config']['activation']
    dropouts = network['layer_config']['dropout']

    model = Sequential()

    for layer in range(n_layers):
        # configure first layer with input_shape
        if layer == 0:
            model.add(
                Dense(n_nodes[layer], input_shape=[input_shape]))
        else:
            model.add(Dense(n_nodes[layer]))

        if (layer_types[layer] == 'batch_norm' or layer_types[layer] == 'batch_norm_dropout'):
            model.add(BatchNormalization())

        model.add(Activation(activations[layer]))

        if (layer_types[layer] == 'dropout' or layer_types[layer] == 'batch_norm_dropout'):
            model.add(Dropout(dropouts[layer]))

    model.add(Dense(output_shape))

    model.compile(optimizer=Adam(amsgrad=True),
                  metrics=['mse', 'mae', MaxAE],
                  loss='mae')
    return model
