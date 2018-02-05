from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from dataset import Dataset
from time import time
import numpy as np

img_width = 100
img_height = 100
file_weights = 'Weights.h5'
        
def build_model():
    # Feature Extraction Layer
    inputs = Input(shape=(100, 100, 3))
    conv_layer1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    max_pooling1 = MaxPooling2D((2, 2))(conv_layer1)
    conv_layer2 = Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D((2, 2))(conv_layer2)

    # Flatten feature map to Vector.
    flatten = Flatten()(max_pooling2)

    # Fully Connected Layer
    fc_layer1 = Dense(1000, activation='relu')(flatten)
    fc_layer2 = Dense(100, activation='softmax')(fc_layer1)
    outputs = Dense(3, activation='softmax')(fc_layer2)

    return Model(inputs=inputs, outputs=outputs)

def train(path):
    model = build_model()

    # Adam Optimizer and Cross Entropy Loss
    sgd = SGD(lr=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    #print model summary
    print(model.summary())

    # Use TensorBoard
    callbacks = TensorBoard(log_dir='./Graph/{}'.format(time()), histogram_freq=0, write_graph=True, write_images=True)

    epochs = 50
    batch_size = 16
    train_data = Dataset(path, img_width, img_height)
    x_train, y_train, nama_file, nama_label = train_data.getData()


    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callbacks])

    # Save Weights
    model.save_weights(file_weights)

def prediksi(path):
    model = build_model()

    # Adam Optimizer and Cross Entropy Loss
    sgd = SGD(lr=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy'])

    model.load_weights(file_weights)

    test_data = Dataset(path, img_width, img_height)
    x_test, y_test, nama_file, nama_label = test_data.getData()

    classes = model.predict(x_test)

    return classes, nama_label, y_test, nama_file


if __name__ == '__main__':
    classes, nama_label, y_test, nama_file = prediksi('data/test')
    for i, j in zip(classes, y_test):
        status = '\033[1;31m Salah \033[0m'
        if np.where(i==max(i)) == np.where(j==max(j)):
            status = '\033[1;32m Benar \033[0m'
        print('{0}  |   {1} |   {2}'.format(i,j,status))
