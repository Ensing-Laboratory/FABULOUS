from keras import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Lambda, add
from keras.optimizers import Adam
import keras.backend as K
from keras.backend import clear_session


def build_model(network):
    clear_session()

    def MaxAE(y_true, y_pred):
        minus_y_pred = Lambda(lambda x: -x)(y_pred)
        return K.max(K.abs(add([y_true, minus_y_pred])))

    n_layers = network['network_config']['n_layers']
    n_nodes = network['layer_config']['n_nodes']

    model = Sequential()
    layer_types = network['layer_config']['layer_type']

    for layer in range(network['network_config']['n_layers']):
        # configure first layer with input_shape
        if layer == 0:
            model.add(
                Dense(network['layer_config']['n_nodes'][layer], input_shape=[network['io_config']['input_shape']]))
        else:
            model.add(Dense(network['layer_config']['n_nodes'][layer]))

        if (layer_types[layer] == 'batch_norm' or layer_types[layer] == 'batch_norm_dropout'):
            model.add(BatchNormalization())

        model.add(Activation(network['layer_config']['activation'][layer]))

        if (layer_types[layer] == 'dropout' or layer_types[layer] == 'batch_norm_dropout'):
            model.add(Dropout(network['layer_config']['dropout'][layer]))

    model.add(Dense(network['io_config']['output_shape']))

    model.compile(optimizer=Adam(amsgrad=True),
                  metrics=['mse', 'mae', MaxAE],
                  loss='mae')
    return model
