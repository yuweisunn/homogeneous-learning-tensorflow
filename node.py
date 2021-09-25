def build_edge_discriminator():
    opt = Adam(learning_rate =0.001)

    model = Sequential()
    model.add(InputLayer(input_shape=img_shape))
    model.add(Conv2D(filters=20, kernel_size=5, strides=(1, 1), activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=50, kernel_size=5, strides=(1, 1), activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=10, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
