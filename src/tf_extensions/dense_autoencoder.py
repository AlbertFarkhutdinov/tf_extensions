import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input, Reshape
from keras.models import Model


def create_dense_ae(
    img_shape: tuple[int, ...],
) -> tuple[Model, Model, Model]:
    encoding_dim = 49

    input_img = Input(shape=img_shape)
    encoded = Dense(encoding_dim, activation='relu')(Flatten()(input_img))

    input_encoded = Input(shape=(encoding_dim,))
    decoded = Reshape(img_shape)(
        Dense(
            img_shape[0] * img_shape[1],
            activation='sigmoid',
        )(input_encoded),
    )

    encoder = Model(
        input_img,
        encoded,
        name='encoder',
    )
    decoder = Model(
        input_encoded,
        decoded,
        name='decoder',
    )
    autoencoder = Model(
        input_img,
        decoder(encoder(input_img)),
        name='autoencoder',
    )
    return encoder, decoder, autoencoder


def run():
    img_shape = (28, 28, 1)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    uint8_max = np.iinfo(np.uint8).max
    x_train = x_train.astype('float32') / uint8_max
    x_test = x_test.astype('float32') / uint8_max
    x_train = np.reshape(
        x_train,
        newshape=(len(x_train), *img_shape),
    )
    x_test = np.reshape(
        x_test,
        newshape=(len(x_test), *img_shape),
    )

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print(img_shape)

    encoder, decoder, autoencoder = create_dense_ae(img_shape=img_shape)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()

    autoencoder.fit(
        x_train,
        x_train,
        batch_size=256,
        epochs=50,
        shuffle=True,
        validation_data=(x_test, x_test),
    )
    batch_size = 10
    images = x_test[:batch_size]
    encoded_images = encoder.predict(images, batch_size=batch_size)
    decoded_images = decoder.predict(encoded_images, batch_size=batch_size)
    return encoded_images, decoded_images


if __name__ == '__main__':
    run()
