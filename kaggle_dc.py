import os
import shutil

import h5py
from keras import Input
from keras.applications import ResNet50, InceptionV3, Xception, inception_v3, xception
from keras.engine import Model
from keras.layers import Lambda, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


def process_image():
    train_filenames = os.listdir('./data/image/train')
    train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)
    train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)

    def rmrf_mkdir(dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)

    rmrf_mkdir('train2')
    os.mkdir('train2/cat')
    os.mkdir('train2/dog')
    rmrf_mkdir('test2')
    # os.symlink('./data/image/test1', 'test2/test')
    # shutil.copyfile('./data/image/test1', 'test2/test')
    for filename in train_cat:
        # os.symlink('./data/image/train/' + filename, 'train2/cat/' + filename)
        shutil.copyfile('./data/image/train/' + filename, 'train2/cat/' + filename)
    for filename in train_dog:
        # os.symlink('./data/image/train/' + filename, 'train2/dog/' + filename)
        shutil.copyfile('./data/image/train/' + filename, 'train2/dog/' + filename)


def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("train2", image_size, shuffle=False,
                                              batch_size=16)
    test_generator = gen.flow_from_directory("./data/image/test1", image_size, shuffle=False,
                                             batch_size=16, class_mode=None)

    train = model.predict_generator(train_generator, train_generator.samples)
    test = model.predict_generator(test_generator, test_generator.samples)

    with h5py.File("gap_%s.h5" % MODEL.func_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


write_gap(ResNet50, (224, 224))
write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap(Xception, (299, 299), xception.preprocess_input)

# process_image()
