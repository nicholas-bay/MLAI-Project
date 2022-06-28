from PIL import Image, ImageOps
from pathlib import Path
import tensorflow as tf
import numpy as np

IMAGE_NAME = "test.jpg"

def import_and_predict(image_data, model):
  size = (100, 100)
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  image = image.convert('RGB')
  image = np.asarray(image)
  image = (image.astype(np.float32) / 255.0)
  img_reshape = image[np.newaxis, ...]
  prediction = model.predict(img_reshape)
  return prediction

label = ''
frame = None
model = tf.keras.models.load_model('model.hdf5')
image = Image.open(IMAGE_NAME)
prediction = import_and_predict(image, model)
if np.argmax(prediction) == 0:
  predict = "onion"
elif np.argmax(prediction) == 1:
  predict = "orange"
else:
  predict = "unknown"
print(predict)