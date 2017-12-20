# -*- coding:utf-8 -*-
from keras import Sequential
from keras.applications import InceptionV3
from keras.layers import Flatten, Dense, Dropout
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

IM_WIDTH, IM_HEIGHT = 299, 299
BATCH_SIZE = 64


def bottleneck_feature():
    model = InceptionV3(weights='imagenet', include_top=False)

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    t_generator = datagen.flow_from_directory(
        '../dataset/train_dir',
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)

    v_generator = datagen.flow_from_directory(
        '../dataset/val_dir',
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(t_generator, len(t_generator.filenames) // BATCH_SIZE)
    print(bottleneck_features_train.shape)
    np.save('../models/bottleneck_features_train.npy', bottleneck_features_train)

    bottleneck_features_validation = model.predict_generator(v_generator, len(v_generator.filenames) // BATCH_SIZE)
    print(bottleneck_features_validation.shape)
    np.save('../models/bottleneck_features_validation.npy', bottleneck_features_validation)


def train():
    train_data = np.load('../models/bottleneck_features_train.npy')
    train_labels = np.array([0] * 932 + [1] * 932)

    validation_data = np.load('../models/bottleneck_features_validation.npy')
    validation_labels = np.array([0] * 192 + [1] * 192)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_data, train_labels,
              epochs=50,
              batch_size=BATCH_SIZE,
              validation_data=(validation_data, validation_labels))
    model.save('../models/bottleneck_fc_model.h5')


def predict():
    model = load_model("../models/bottleneck_fc_model.h5")
    # 保存模型结构成图片
    # plot_model(model, to_file='../models/Inception_v3_ft.png')
    lables = {0: 'cat', 1: 'dog'}
    root = '../dataset/test'
    list_dir = os.listdir(root)
    for i in range(0, len(list_dir)):
        path = os.path.join(root, list_dir[i])
        if os.path.isfile(path):
            img = image.load_img(path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = x.astype('float32')
            # x = x.reshape((1,) + x.shape)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            pre = model.predict(x)
            rr = np.argmax(pre, axis=1)[0]
            lable = lables[rr]
            print(path, '====>', pre, lable)


if __name__ == '__main__':
    # bottleneck_feature()
    train()
    # predict()
