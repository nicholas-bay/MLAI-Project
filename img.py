import cv2, sys, time, os
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

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
print("Choose Option:\n1. Live Detection\n2. Static Detection\n3. Add Data")
type = int(input())
if (type == 1):
  while (True):
    ret, original = cap.read()
    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename = 'data/img.jpg', img = original)
    prediction = import_and_predict(Image.open('data/img.jpg'), model)
    if np.argmax(prediction) == 0:
      predict = "onion"
    elif np.argmax(prediction) == 1:
      predict = "orange"
    else:
      predict = "unknown"
    cv2.putText(
      original, predict, (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Live Detection", original)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
      break
  cap.release()
  frame = None
  cv2.destroyAllWindows()
elif (type == 2):
  prediction = import_and_predict(Image.open("data/test.jpg"), model)
  if np.argmax(prediction) == 0:
    predict = "onion"
  elif np.argmax(prediction) == 1:
    predict = "orange"
  else:
    predict = "unknown"
  print(predict)
elif (type == 3):
  print("Type:\n1. Onion\n2. Orange\n3. Unknown")
  obj = int(input())
  while (True):
    time.sleep(0.005)
    if (obj == 1):
      obj_name = "onion"
      index = 0
    elif (obj == 2):
      obj_name = "orange"
      index = 1
    if (obj == 3):
      obj_name = "unknown"
      index = 2

    with open('data/count.txt', 'r') as file:
      data = file.readlines()
    count = data[index][:-1]
    ret, original = cap.read()
    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename = 'data/img.jpg', img = original)
    Path("data/img.jpg").rename("data/" + obj_name + "/" + count + ".jpg")
    data[index] = str(int(count) + 1) + "\n"
    with open('data/count.txt', 'w') as file:
      file.writelines(data)
    cv2.imshow("Add Data", original)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
      break
  cap.release()
  frame = None
  cv2.destroyAllWindows()
os.remove("data/img.jpg")
sys.exit()