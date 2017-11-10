"""
采用函数式模式实现CNN网络
"""
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import np_utils
import numpy as np

img_rows, img_cols = 28, 28
nb_classes = 10
epochs = 12
pool_size = (2, 2)
# 卷积核的大小
kernel_size = (3, 3)
nb_filters = 32
(X_train, y_train), (X_test, y_test) = mnist.load_data()

img = image.load_img("D:\\Demo\\python\\data\\image\\4.png", target_size=(28, 28))
img_array = image.img_to_array(img)
img_array = img_array.reshape((3, 28, 28))
# print(X_train.shape)
# print(img_array.shape)
cc = np.vstack((X_train, img_array))
# X_train = cc
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
# 将X_train, X_test的数据格式转为float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化
X_train /= 255
X_test /= 255

# 转换为类别标签
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

inputs = Input(shape=(img_rows, img_cols, 1))

x = Conv2D(nb_filters, kernel_size, padding='same', activation='relu')(inputs)
x = MaxPooling2D(pool_size)(x)

x = Conv2D(64, kernel_size, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size, strides=(1, 1), padding='same')(x)

x = Conv2D(128, kernel_size, padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size, strides=(1, 1), padding='same')(x)

x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
sgd = Adam(lr=0.001)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, Y_train, batch_size=128, epochs=10)
model.fit(X_train, Y_train, batch_size=128, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save("d:\\demo\\python\models\\cnn_mnist_function_1.h5")
