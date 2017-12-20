from keras import Sequential, Input, Model
from keras.layers import Embedding, Flatten, Reshape, Dense
import numpy as np
import pandas as pd

# model = Sequential()
# model.add(Embedding(1000, 64, input_length=10))
# model.add(Flatten())
# # the model will take as input an integer matrix of size (batch, input_length).
# # the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# # now model.output_shape == (None, 10, 64), where None is the batch dimension.
# input_array = np.random.randint(1000, size=(32, 10))
# # print(input_array.shape)
#
# model.compile('rmsprop', 'mse')
# output_array = model.predict(input_array)
# print(output_array.shape)
# assert output_array.shape == (32, 10, 64)
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.specs.python import Relu


def cross_columns(x_cols):
    """simple helper to build the crossed columns in a pandas dataframe
    """
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    # print(x_cols)
    # print(colnames)
    for x, y, z in zip(colnames, x_cols, [1, 2, 3]):
        print(x, '|', y, z)
    for cname, x_c in zip(colnames, x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns


x_cols = (['education', 'occupation'], ['native_country', 'occupation'])
x = cross_columns(x_cols)
# print(x)


if __name__ == '__main__':
    inp = Input(shape=(1,), dtype='float32')
    out = Dense(1, activation='relu')(inp)
    # print(pd.__version__)
    m = Model(inp, out)
    m.compile('rmsprop', 'mse')
    i = np.random.rand(10, 1)
    print(i)
    r = m.predict(i)
    print(r)
