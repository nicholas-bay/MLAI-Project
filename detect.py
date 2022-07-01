import cv2, sys
from PIL import Image, ImageOps
from pathlib import Path
import tensorflow as tf
import numpy as np

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

cap = cv2.VideoCapture(0)
if (cap.isOpened()):
  print("Camera OK")
else:
  cap.open()

while (True):
  ret, original = cap.read()
  frame = cv2.resize(original, (224, 224))
  cv2.imwrite(filename = 'img.jpg', img = original)
  image = Image.open('img.jpg')
  prediction = import_and_predict(image, model)
  if np.argmax(prediction) == 0:
    predict = "onion"
  elif np.argmax(prediction) == 1:
    predict = "orange"
  else:
  # with open('data/unknown_count.txt', 'r') as file:
  #   count = file.read()
  # Path("img.jpg").rename("data/unknown/" + count + ".jpg")
  # test = int(count) + 1
  # with open('data/unknown_count.txt', 'w') as file:
  #   file.write(str(test))
    predict = "unknown"
  cv2.putText(
    original, predict, (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
  cv2.putText(
    original, str(prediction), (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
  cv2.imshow("Detector", original)

  if (cv2.waitKey(1) & 0xFF == ord('q')):
    break

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()