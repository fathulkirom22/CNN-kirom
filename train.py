from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from dataset import Dataset
import numpy as np
from time import time

# Feature Extraction Layer
inputs = Input(shape=(100, 100, 3))
conv_layer1 = Conv2D(32, (5, 5), strides=(1,1), activation='relu')(inputs)
max_pooling1 = MaxPooling2D((3, 3))(conv_layer1)
conv_layer2 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu')(max_pooling1)
max_pooling2 = MaxPooling2D((3, 3))(conv_layer2)

# Flatten feature map to Vector.
flatten = Flatten()(max_pooling2)

# Fully Connected Layer
fc_layer1 = Dense(1000, activation='relu')(flatten)
fc_layer2 = Dense(100, activation='softmax')(fc_layer1)
outputs = Dense(3, activation='softmax')(fc_layer2)

model = Model(inputs=inputs, outputs=outputs)

# Adam Optimizer and Cross Entropy Loss
sgd = SGD(lr=0.0001)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Print Model Summary
print(model.summary())

#----------------------------------------------------------------------------------------------------------------------------

# Use TensorBoard
callbacks = TensorBoard(log_dir='./Graph/{}'.format(time()), histogram_freq=0, write_graph=True, write_images=True)

train_data_dir = Dataset('data/train', 100)
#validation_data_dir = 'data/validation'
epochs = 10
batch_size = 32
img_width, img_height = 100, 100
x_train, y_train = train_data_dir.getData()


model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size = batch_size,
    callbacks=[callbacks])

# Save Weights
model.save_weights('weights.h5')


