from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard
from dataset import Dataset
from time import time
import numpy as np
import sys
import getopt

img_width = 100
img_height = 100
epochs = 50
batch_size = 16
learning_rate = 0.0001
path_train = 'data/train'
path_test = 'data/test'
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

    model = Model(inputs=inputs, outputs=outputs)

    # SGD Optimizer and Cross Entropy Loss
    sgd = SGD(lr=learning_rate)
    model.compile(
        optimizer=sgd, 
         loss='categorical_crossentropy',
          metrics=['accuracy'])
        
    return model

def train():
    model = build_model()

    #print model summary
    print(model.summary())

    # Use TensorBoard
    callbacks = TensorBoard(
        log_dir='./Graph/{}'.format(time()),
         histogram_freq=0,
          write_graph=True,
           write_images=True)

    train_data = Dataset(path_train, img_width, img_height)
    x_train, y_train, nama_file, nama_label = train_data.getData()

    model.fit(
        x_train,
         y_train,
          epochs=epochs,
           batch_size=batch_size,
            callbacks=[callbacks])

    # Save Weights
    model.save_weights(file_weights)

def test():
    model = build_model()

    model.load_weights(file_weights)

    test_data = Dataset(path_test, img_width, img_height)
    x_test, y_test, nama_file, nama_label = test_data.getData()

    classes = model.predict(x_test)

    return classes, y_test, nama_file, nama_label

def main(opt):
    if opt == '--help':
        print('CNN_kirom.py [--help|--train|--test]')
        sys.exit()
    elif opt == '--train':
        train()
    elif opt == '--test':
        classes, y_test, nama_file, nama_label = test()
        for i, j in zip(classes, y_test):
            status = '\033[1;31m Salah \033[0m'
            if np.where(i == max(i)) == np.where(j == max(j)):
                status = '\033[1;32m Benar \033[0m'
            print('{0}  |   {1} |   {2}'.format(i, j, status))
    else:
        print('CNN_kirom.py [--help|--train|--test]')
        sys.exit()

if __name__ == '__main__':
    main(sys.argv[1])
