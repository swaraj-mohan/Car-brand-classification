#import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from glob import glob

#resize all images to the expected size
image_size = [224, 224]

#set train, test, validation dataset path
train_path = '/content/drive/My Drive/Datasets/Car brand classification/Datasets/Train'
test_path = '/content/drive/My Drive/Datasets/Car brand classification/Datasets/Test'

#import the ResNet50 architecture and add preprocessing layer, we are using ImageNet weights
ResNet50_model = keras.applications.resnet50.ResNet50(input_shape = image_size + [3], weights = 'imagenet', include_top = False)

#freeze the weights of the pre-trained layers
for layer in ResNet50_model.layers:
  layer.trainable = False
  
#useful for getting number of output classes
folders = glob('/content/drive/My Drive/Datasets/Car brand classification/Datasets/Train/*')

#adding our own layers
layer_flatten = keras.layers.Flatten()(ResNet50_model.output)
output = keras.layers.Dense(len(folders), activation = "softmax")(layer_flatten)
model = keras.Model(inputs = ResNet50_model.input, outputs = output)

#summary of our model
print(model.summary())

#compile the model and specify loss function and optimizer
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

#use the ImageDataGenerator class to load images from the dataset
train_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

#make sure you provide the same target size as initialied for the image size
training_set = train_data_generator.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_data_generator.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

#train the model
history = model.fit_generator(
  training_set,
  validation_data = test_set,
  epochs = 50,
  steps_per_epoch = len(training_set),
  validation_steps = len(test_set)
)

#save the model as an h5 file
model.save('/content/drive/My Drive/Datasets/Car brand classification/model_resnet50.h5')

#plot the loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#plot the accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#evaluate the model
print(model.evaluate(test_set))

#using the model to make predictions
y_pred = model.predict(test_set)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)