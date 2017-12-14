from keras import Model, Input, layers
from keras.initializers import RandomNormal
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Flatten, Activation, BatchNormalization, LeakyReLU, \
    Dropout, Conv2DTranspose, MaxPooling2D, regularizers, multiply, Embedding
from keras.layers import Reshape
# from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    # model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def create_D():
    # weights are initlaized from normal distribution with below params
    weight_init = RandomNormal(mean=0., stddev=0.02)

    input_image = Input(shape=(28, 28, 1), name='input_image')

    x = Conv2D(
        32, (3, 3),
        padding='same',
        name='conv_1',
        kernel_initializer=weight_init)(input_image)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(
        64, (3, 3),
        padding='same',
        name='conv_2',
        kernel_initializer=weight_init)(x)
    x = MaxPool2D(pool_size=1)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(
        128, (3, 3),
        padding='same',
        name='conv_3',
        kernel_initializer=weight_init)(x)
    x = MaxPool2D(pool_size=2)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(
        256, (3, 3),
        padding='same',
        name='coonv_4',
        kernel_initializer=weight_init)(x)
    x = MaxPool2D(pool_size=1)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    features = Flatten()(x)

    output_is_fake = Dense(
        1, activation='linear', name='output_is_fake')(features)

    output_class = Dense(
        10, activation='softmax', name='output_class')(features)

    return Model(
        inputs=[input_image], outputs=[output_is_fake], name='D')


def create_G(latent_size=100):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    cnn = Sequential()

    cnn.add(Dense(3 * 3 * 384, input_dim=latent_size, activation='relu'))
    cnn.add(Reshape((3, 3, 384)))

    # upsample to (7, 7, ...)
    cnn.add(Conv2DTranspose(192, 5, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal'))

    # upsample to (14, 14, ...)
    cnn.add(Conv2DTranspose(96, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal'))

    # upsample to (28, 28, ...)
    cnn.add(Conv2DTranspose(1, 5, strides=2, padding='same',
                            activation='tanh',
                            kernel_initializer='glorot_normal'))

    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size,))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    cls = Flatten()(Embedding(10, latent_size,
                              embeddings_initializer='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    return Model(latent, fake_image)


def discriminator_model():
    # model = Sequential()
    # model.add(Conv2D(64, (5, 5),
    #                  padding='same',
    #                  input_shape=(28, 28, 1)))
    # model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, (5, 5)))
    # model.add(Activation('tanh'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(1024))
    # model.add(Activation('tanh'))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    # return model

    weight_init = RandomNormal(mean=0., stddev=0.02)

    input_image = Input(shape=(28, 28, 1), name='input_image')

    x = Conv2D(
        32, (3, 3),
        padding='same',
        name='conv_1',
        kernel_initializer=weight_init)(input_image)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(
        64, (3, 3),
        padding='same',
        name='conv_2',
        kernel_initializer=weight_init)(x)
    x = MaxPool2D(pool_size=1)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(
        128, (3, 3),
        padding='same',
        name='conv_3',
        kernel_initializer=weight_init)(x)
    x = MaxPool2D(pool_size=2)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(
        256, (3, 3),
        padding='same',
        name='coonv_4',
        kernel_initializer=weight_init)(x)
    x = MaxPool2D(pool_size=1)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    features = Flatten()(x)

    output_is_fake = Dense(
        1, activation='linear', name='output_is_fake')(features)

    output_class = Dense(
        10, activation='softmax', name='output_class')(features)

    return Model(
        inputs=[input_image], outputs=[output_is_fake, output_class], name='D')


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def combine_images(generated_images):
    # (256,1,28, 28)
    num = generated_images.shape[0]
    width = int(math.sqrt(num))  # 16
    height = int(math.ceil(float(num) / width))  # 16
    shape = generated_images.shape[2:]  # 28x28
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)  # 生成448x448的图片
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image


def train(BATCH_SIZE):
    """
    生成器先生成图片，训练判别器，训练对抗网络，得到loss
    :param BATCH_SIZE:
    :return:
    """

    discriminator = create_D()

    generator = create_G()

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # print(X_train.shape)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]

    # discriminator = discriminator_model()
    # generator = generator_model()
    # discriminator = Discriminator()()
    # generator = Generator()()

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    # d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    d_optim = Adam(lr=0.0005)
    g_optim = Adam(lr=0.0005)
    generator.compile(loss='binary_crossentropy', optimizer="adam")
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):  # 训练图片总数/BATACH_SIZE
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)  # 生成正太分布的随机数
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]  # 每次取batch_size的图片ndarray
            # image_batch = image_batch.transpose((0, 2, 3, 1))  # 最后一个维度为补充的维度，前面三个维度为图片本身的维度属性
            generated_images = generator.predict(noise, verbose=0)  # 一次生成256张28x28的图片

            print("generated_images shape:", generated_images.shape)
            if index % 40 == 0:  # 每20次输出一次图片
                generated_images_tosave = generated_images.transpose((0, 3, 1, 2))  # 256 1,28,28
                # print("save images shape:", generated_images_tosave.shape)
                image = combine_images(generated_images_tosave)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(str(epoch) + "_" + str(index) + ".png")
            print(image_batch.shape, generated_images.shape)
            X = np.concatenate((image_batch, generated_images))

            # 把训练图片和生成的图片合并在一起
            # if index % 64 == 0:
            #     x_tosave = X.transpose((0, 3, 1, 2))
            #     image = combine_images(x_tosave)
            #     image = image * 127.5 + 127.5
            #     Image.fromarray(image.astype(np.uint8)).save(
            #         str(epoch) + "_" + str(index) + "_combine.png")

            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE  # 标签0为生成器生成的图片
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator.h5', True)
                discriminator.save_weights('discriminator.h5', True)


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator.h5')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator.h5')
        noise = np.zeros((BATCH_SIZE * 20, 100))
        for i in range(BATCH_SIZE * 20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        generated_images_tosave = generated_images.transpose((0, 3, 1, 2))
        image = combine_images(generated_images_tosave)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train(BATCH_SIZE=256)
    # generate(BATCH_SIZE=256, nice=False)
    # args = get_args()
    # if args.mode == "train":
    #     train(BATCH_SIZE=args.batch_size)
    # elif args.mode == "generate":
    #     generate(BATCH_SIZE=args.batch_size, nice=args.nice)
