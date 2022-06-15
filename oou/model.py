# importing dependancies
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# setting up env
image_size = (100, 100)
ds = os.path.join('data/')
train_dir = os.path.join('data/train/')
test_dir = os.path.join('data/test/')
valid_dir = os.path.join('data/valid/')
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  image_size = image_size,
  batch_size = 50)
val_ds = tf.keras.utils.image_dataset_from_directory(
  valid_dir,
  image_size = image_size,
  batch_size = 10)
test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  image_size = image_size,
  batch_size = 10)

# print class_names
class_names = train_ds.class_names
print(class_names)

# visualing data
plt.figure(figsize = (10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype('uint8'))
    plt.title(class_names[labels[i]])
    plt.axis('off')
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  
# normaliszation-train
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y:(normalization_layer(x), y))
image_batch, labels_batch = next(iter(train_ds))
first_image = image_batch[0]

# normaliszation-valid
normalization_layer = tf.keras.layers.Rescaling(1./255)
val_ds = val_ds.map(lambda x, y:(normalization_layer(x), y))
image_batch, labels_batch = next(iter(val_ds))
first_image = image_batch[0]

# notice pixal values are between 0 and 1
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

# model
num_classes = 3
input_data = tf.keras.layers.Input([100, 100, 3])

# main layer 
conv = tf.keras.layers.Conv2D(30, 3, activation = 'relu')(input_data)
conv = tf.keras.layers.Conv2D(30, 3, activation = 'relu')(conv)
conv = tf.keras.layers.MaxPooling2D()(conv)
conv = tf.keras.layers.Conv2D(30, 3, activation = 'relu')(conv)
conv = tf.keras.layers.Conv2D(30, 3, activation = 'relu')(conv)
conv = tf.keras.layers.MaxPooling2D()(conv)
dense = tf.keras.layers.Flatten()(conv)
output_data = tf.keras.layers.Dense(num_classes)(dense)
model = tf.keras.Model(inputs = input_data, outputs = output_data)
model.summary()

# compiling and optimizer
model.compile(
  optimizer = 'adam',
  loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True),
  metrics = ['accuracy'])

# fit data
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=10)
print("Now testing the model:")
model.evaluate(test_ds)
tf.keras.models.save_model(model, 'model.hdf5')