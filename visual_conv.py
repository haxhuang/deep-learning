import matplotlib.pyplot as plt
from keras.layers import Convolution2D, Activation, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing import image
import numpy as np

file = "D:\\Demo\\python\\data\\image\\cat.png"
img = image.load_img(file)
x = image.img_to_array(img)
# print(x.shape)
model = Sequential()
model.add(Convolution2D(3, (3, 3), input_shape=x.shape))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# for t in range(3):
#     model.add(Convolution2D(1, (3, 3), input_shape=x.shape))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

cat_batch = np.expand_dims(img, axis=0)
conv_cat = model.predict(cat_batch)
# print(conv_cat.shape)


def show_cat(cat_batch):
    cat = np.squeeze(cat_batch, axis=0)
    print(cat.shape)
    plt.imshow(cat)
    plt.show()


def nice_cat_printer(model, cat):
    cat_batch = np.expand_dims(cat, axis=0)
    conv_cat2 = model.predict(cat_batch)
    conv_cat2 = np.squeeze(conv_cat2, axis=0)
    print(conv_cat2.shape)
    conv_cat2 = conv_cat2.reshape(conv_cat2.shape[:2])
    print(conv_cat2.shape)
    plt.imshow(conv_cat2)
    plt.show()


# nice_cat_printer(model, x)
show_cat(conv_cat)
