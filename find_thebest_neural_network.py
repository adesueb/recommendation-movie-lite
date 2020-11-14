import time

dense_layers = [0, 1, 2, 3]
lstms = [1, 2, 3, 4]
layer_sizes = [64, 128, 256, 512]

for dense_layer in dense_layers:
    for lstm in lstms:
        for layer_size in layer_sizes:
            name = "{}-nodes-{}-lstm-{}-dense-{}".format(layer_size, lstm, dense_layer, int(time.time()))
            print(name)

            # tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

            # model = Sequential()
            # model.add(Embedding(lenSortedClassContents, layer_size, input_length=MAX_SEQUENCE))

            # for i in range(lstm):
            #     model.add(Bidirectional(LSTM(layer_size)))

            # for i in range(dense_layer):
            #     model.add(Dense(layer_size))
            #     model.add(Activation('relu'))

            # model.add(Dense(lenSortedClassContents, activation='softmax'))
            # adam = Adam(lr=0.01)
            # model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            # history = model.fit(X, Y, epochs=100, verbose=1, callbacks=[tensorboard])
