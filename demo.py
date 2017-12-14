from PIL import Image
from keras.applications import VGG16, InceptionV3
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense



# （1）载入图片
# 图像生成器初始化
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

np.math.sqrt(256)
model = InceptionV3(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1. / 255)

# 训练集图像生成器
generator = datagen.flow_from_directory(
    '../dataset/train_dir',
    target_size=(299, 299),
    batch_size=64,
    class_mode=None,
    shuffle=False)

# 　验证集图像生成器
generator = datagen.flow_from_directory(
    '../dataset/val_dir',
    target_size=(299, 299),
    batch_size=64,
    class_mode=None,
    shuffle=False)

# （2）灌入pre-model的权重
# model.load_weights('../models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# （3）得到bottleneck feature
bottleneck_features_train = model.predict_generator(generator, 32)
# 核心，steps是生成器要返回数据的轮数，每个epoch含有500张图片，与model.fit(samples_per_epoch)相对
np.save('../models/bottleneck_features_train.npy', bottleneck_features_train)

bottleneck_features_validation = model.predict_generator(generator, 6)
# 与model.fit(nb_val_samples)相对，一个epoch有800张图片，验证集
np.save('../models/bottleneck_features_validation.npy', bottleneck_features_validation)
