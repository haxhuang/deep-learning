from keras import callbacks
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, Input, BatchNormalization
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras.utils.vis_utils import model_to_dot
import tensorflow as tf
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from IPython.display import SVG
from genplate import *
from keras.preprocessing import image

np.random.seed(5)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z"
         ];

M_strIdx = dict(zip(chars, range(len(chars))))

n_generate = 100
rows = 20
cols = int(n_generate / rows)

G = GenPlate("./font/platech.ttf", './font/platechar.ttf', "./NoPlates")
l_plateStr, l_plateImg = G.genBatch(100, 2, range(31, 65), "./plate", (272, 72))

l_out = []
for i in range(rows):
    l_tmp = []
    for j in range(cols):
        l_tmp.append(l_plateImg[i * cols + j])

    l_out.append(np.hstack(l_tmp))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(np.vstack(l_out), aspect="auto")
    # plt.show()


def gen(batch_size=32):
    while True:
        l_plateStr, l_plateImg = G.genBatch(batch_size, 2, range(31, 65), "./plate", (272, 72))
        X = np.array(l_plateImg, dtype=np.uint8)
        # X = X / 255
        ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], l_plateStr)), dtype=np.uint8)
        y = np.zeros([ytmp.shape[1], batch_size, len(chars)])
        for batch in range(batch_size):
            for idx, row_i in enumerate(ytmp[batch]):
                y[idx, batch, row_i] = 1

        yield X, [yy for yy in y]  # 有7个outputs，所以这里需要转变成维度为7的数组lables


def train():
    global i
    batch_size = 64
    adam = Adam(lr=0.001)
    input_tensor = Input((72, 272, 3))
    x = input_tensor

    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)

    n_class = len(chars)
    x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(7)]
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    SVG(model_to_dot(model=model, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))
    best_model = ModelCheckpoint("./models/plate_reg_best.h5", monitor='val_loss', verbose=0, save_best_only=True)
    tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    model.fit_generator(gen(batch_size), steps_per_epoch=600, epochs=35,
                        validation_data=gen(batch_size), validation_steps=100,
                        callbacks=[best_model, tensorboard]
                        )
    model.save('./models/plate_reg.h5')


def predict():
    model = load_model('./models/plate_reg.h5')
    # model.save_weights('./models/plate_reg.w')
    rootpath = './predict/'
    list_dir = os.listdir(rootpath)
    for i in range(0, len(list_dir)):
        path = os.path.join(rootpath, list_dir[i])
        if os.path.isfile(path):
            # cv2读取中文路径会报错，需要处理下
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)  # 解决中文路径读取报错
            img = cv2.resize(img, (272, 72))
            x = np.array([img], dtype=np.uint8)
            l_titles = list(
                map(lambda x1: "".join([chars[xx] for xx in x1]), np.argmax(np.array(model.predict(x)), 2).T))
            print(path, l_titles)


if __name__ == '__main__':
    # train()
    predict()
