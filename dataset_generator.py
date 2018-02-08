from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

path = 'data/train'

def runImageGenerator(path_img, save_in):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    img = load_img(path_img)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    # this is a Numpy array with shape (1, 3, 150, 150)
    x = x.reshape((1,) + x.shape)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='preview/{}'.format(save_in),
                               save_prefix='matang',
                                save_format='jpeg'):
        i += 1
        if i > 50:
            break  # otherwise the generator would loop indefinitely

def getLabel():
    #only directory
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def getDataNameFile():
    tmpReturn = []
    for i in getLabel():
      newPath = path+'/'+i
      #only file
      tmpReturn.append([name for name in os.listdir(newPath)
                        if os.path.isfile(os.path.join(newPath, name))])
    return tmpReturn

def main():
    tmpLabel = getLabel()
    index = 0
    for i in getDataNameFile():
        tmp = []
        for j in i:
            save_in = tmpLabel[index]
            path_img = '{0}/{1}/{2}'.format(path, save_in, j)
            runImageGenerator(path_img, save_in)
        index += 1

if __name__ == '__main__':
    main()