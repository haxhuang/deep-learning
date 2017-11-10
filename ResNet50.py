from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

base_model = ResNet50()
img_path = 'D:\\test\\bmw.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
pred = base_model.predict(x)
print('Predicted:', decode_predictions(pred, top=3)[0])