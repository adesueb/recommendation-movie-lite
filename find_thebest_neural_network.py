import time

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def run_hyper_parameter(X, Y, X_val, Y_val, lenSortedClassContents):
    dense_layers = [0, 1, 2, 3, 4]
    layer_sizes = [64, 128, 256, 512]
    optimizers = [0, 1, 2]

    for dense_layer in dense_layers:
        for optimizer in optimizers:
            for layer_size in layer_sizes:
                name = "{}-nodes-{}-optimizer-{}-dense-{}".format(layer_size, optimizer, dense_layer, int(time.time()))
                print(name)

                tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

                model = Sequential()

                model.add(Dense(layer_size, activation="relu", input_shape=X.shape[1:]))

                for i in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(lenSortedClassContents, activation='softmax'))

                if optimizer == 0:
                    opt = Adam(lr=0.01)
                elif optimizer == 1:
                    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
                else:
                    opt = 'adam'
                model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
                model.fit(X, Y, epochs=250, batch_size=64, verbose=1, validation_data=(X_val, Y_val),
                          callbacks=[tensorboard])
