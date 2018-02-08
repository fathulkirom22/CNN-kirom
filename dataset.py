import os
from cv2 import imread, resize, INTER_LINEAR
import numpy as np
from sklearn.utils import shuffle

class Dataset:

  def __init__(self, path, img_width, img_height):
    self.path = path
    self.img_width = img_width
    self.img_height = img_height

  def getLabel(self):
    #only directory
    return [name for name in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, name))]

  def countLabel(self):
    return len(self.getLabel())

  def getDataNameFile(self):
    tmpReturn = []
    for i in self.getLabel():
      newPath = self.path+'/'+i
      #only file
      tmpReturn.append([name for name in os.listdir(newPath) if os.path.isfile(os.path.join(newPath, name))])
    return tmpReturn

  def getData(self):
    datas = []
    labels = []
    name_file = []
    name_label = []

    tmpLabel = self.getLabel()
    index = 0
    for i in self.getDataNameFile():
      tmp = []
      for j in i:
        #datas
        fl = imread('{0.path}/{1}/{2}'.format(self, tmpLabel[index], j))
        fl = resize(fl, (self.img_width, self.img_height), 0, 0, INTER_LINEAR)
        fl = fl.astype(np.float32)
        fl = np.multiply(fl, 1.0 / 255.0)  # change range 0 - 255 to 0 - 1
        datas.append(fl)
        #labels
        label = np.zeros(self.countLabel())
        label[index] = 1.0
        labels.append(label)
        #name file
        name_file.append(j)
        #name class
        name_label.append(tmpLabel[index])
      index += 1

    #numpy array
    datas = np.array(datas)
    labels = np.array(labels)
    name_file = np.array(name_file)
    name_label = np.array(name_label)
    #random
    datas, labels, name_file, name_label = shuffle(datas, labels, name_file, name_label)
    return datas, labels, name_file, name_label
