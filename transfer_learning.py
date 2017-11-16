import os
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np

from keras import callbacks
from keras.applications import VGG16, VGG19
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.utils import plot_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptionV3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
      base_model: keras model excluding top
      nb_classes: # of classes
    Returns:
      new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
      model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    args.train_dir = '../dataset/train_dir'
    args.val_dir = '../dataset/val_dir'
    args.nb_epoch = 5
    args.batch_size = 64
    args.plot = True
    args.output_model_file = '../models/Inception_v3_ft.h5'

    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    # data prep
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        # save_to_dir='../dataset/train_gen',
        batch_size=batch_size, class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size, class_mode='categorical'
    )

    t_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode=None,  # this means our generator will only yield batches of data, no labels
        shuffle=False)  # our data will
    v_generator = train_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        class_mode=None,  # this means our generator will only yield batches of data, no labels
        shuffle=False)  # our data will

    # setup model
    base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, nb_classes)

    bottleneck_features_train = base_model.predict_generator(t_generator, 32)  # (128, 8, 8, 2048)
    print(bottleneck_features_train.shape)
    # np.save(open('../models/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    np.save('../models/bottleneck_features_train.npy', bottleneck_features_train)
    bottleneck_features_validation = base_model.predict_generator(v_generator, 6)
    np.save('../models/bottleneck_features_validation.npy', bottleneck_features_validation)
    print(bottleneck_features_validation.shape)
    # np.save(open('../models/bottleneck_features_validation.npy', 'wb'),
    #         bottleneck_features_validation)

    callback = callbacks.TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=True)

    # transfer learning
    # setup_to_transfer_learn(model, base_model)
    # history_tl = model.fit_generator(
    #     train_generator,
    #     epochs=nb_epoch,
    #     steps_per_epoch=nb_train_samples // batch_size,
    #     validation_data=validation_generator,
    #     validation_steps=nb_val_samples // batch_size,
    #     callbacks=[callback],
    #     class_weight='auto')

    # fine-tuning
    setup_to_finetune(model)
    #
    history_ft = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        callbacks=[callback],
        class_weight='auto')

    # model.evaluate_generator(validation_generator, steps=10)
    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_ft)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


def predict():
    model = load_model("../models/Inception_v3_ft.h5")
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
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            pre = model.predict(x)
            rr = np.argmax(pre, axis=1)[0]
            lable = lables[rr]
            print(path, '====>', pre, lable)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir")
    a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    # if args.train_dir is None or args.val_dir is None:
    #     a.print_help()
    #     sys.exit(1)
    #
    # if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
    #     print("directories do not exist")
    #     sys.exit(1)

    train(args)

    # predict()
