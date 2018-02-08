from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import  to_categorical
from keras.callbacks import TensorBoard
from dataset import Dataset
from time import time
import numpy as np
import sys
import getopt
from sklearn.utils import check_array

img_width = 100
img_height = 100
epochs = 50
batch_size = 16
learning_rate = 0.0001
path_train = 'data/appel/train'
path_test = 'data/appel/test'
file_weights = 'Weights.h5'
# file_weights = 'Weights_save/Weights_feb_5-appel.h5'
        
def build_model(node_output):
    # Feature Extraction Layer
    inputs = Input(shape=(img_width, img_height, 3))
    conv_layer1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    max_pooling1 = MaxPooling2D((2, 2))(conv_layer1)
    conv_layer2 = Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D((2, 2))(conv_layer2)

    # Flatten feature map to Vector.
    flatten = Flatten()(max_pooling2)

    # Fully Connected Layer
    fc_layer1 = Dense(1000, activation='relu')(flatten)
    fc_layer2 = Dense(100, activation='relu')(fc_layer1)
    outputs = Dense(node_output, activation='softmax')(fc_layer2)

    model = Model(inputs=inputs, outputs=outputs)

    # SGD Optimizer and Cross Entropy Loss
    sgd = SGD(lr=learning_rate)
    model.compile(
        optimizer=sgd, 
         loss='categorical_crossentropy',
          metrics=['accuracy'])
        
    return model

def train():
    train_data = Dataset(path_train, img_width, img_height)
    x_train, y_train, nama_file, nama_label = train_data.getData()
    node_output = train_data.countLabel()

    # x_train = to_categorical(x_train)
    # y_train = to_categorical(y_train)

    model = build_model(node_output)

    #print model summary
    print(model.summary())

    # Use TensorBoard
    callbacks = TensorBoard(
        log_dir='./Graph/{}'.format(time()),
         histogram_freq=0,
          write_graph=True,
           write_images=True)

    model.fit(
        x_train,
         y_train,
          epochs=epochs,
           batch_size=batch_size,
            callbacks=[callbacks])

    # Save Weights
    model.save_weights(file_weights)

def test():
    test_data = Dataset(path_test, img_width, img_height)
    x_test, y_test, nama_file, nama_label = test_data.getData()
    node_output = test_data.countLabel()

    # x_train = to_categorical(x_test)
    # y_train = to_categorical(y_test)

    model = build_model(node_output)

    model.load_weights(file_weights)

    classes = model.predict(x_test)

    return classes, y_test, nama_file, nama_label

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main(opt):
    if opt == '--help':
        print('CNN_kirom.py [--help|--train|--test]')
        sys.exit()
    elif opt == '--train':
        train()
    elif opt == '--test':
        classes, y_test, nama_file, nama_label = test()
        count_total = 0
        count_benar = 0
        count_salah = 0
        for i, j in zip(classes, y_test):
            if np.where(i == max(i)) == np.where(j == max(j)):
                status = '\033[1;32m Benar \033[0m'
                count_benar += 1
            else:
                status = '\033[1;31m Salah \033[0m'
                count_salah += 1
            count_total += 1
            # _MAPE = mean_absolute_percentage_error(j,i)
            print('{0}  |   {1} |   {2} |   Error MAPE : - '.format(i, j, status))
        persentase_benar = (count_benar/count_total)*100
        persentase_salah = (count_salah/count_total)*100
        print('benar : {0} | {1}%  salah : {2} | {3}%'.format(count_benar, persentase_benar, count_salah, persentase_salah))
    else:
        print('CNN_kirom.py [--help|--train|--test]')
        sys.exit()

if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except:
        print('CNN_kirom.py [--help|--train|--test]')
