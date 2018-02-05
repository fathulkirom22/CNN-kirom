import os
from cv2 import imread, resize, INTER_LINEAR
import numpy as np
from sklearn.utils import shuffle

class Dataset:

  def __init__(self, path, image_size):
    self.path = path
    self.image_size = image_size

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
    tmpLabel = self.getLabel()
    index = 0
    for i in self.getDataNameFile():
      tmp = []
      for j in i:
        #data
        fl = imread('{0.path}/{1}/{2}'.format(self, tmpLabel[index], j))
        fl = resize(fl, (self.image_size, self.image_size),0, 0, INTER_LINEAR)
        fl = fl.astype(np.float32)
        datas.append(fl)
        #label
        label = np.zeros(self.countLabel())
        label[index] = 1.0
        labels.append(label)
      index += 1

    datas = np.array(datas)
    labels = np.array(labels)
    #random
    datas, labels = shuffle(datas, labels)
    return datas, labels

  

# data = Dataset('data/train', 100)
# print(data.getLabel())
# print(data.countLabel())
# data.getDataNameFile()
# print(data.getData())

