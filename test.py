'''from mnist.loader import MNIST
import random
import numpy as np

mndata = MNIST('data')
features, labels = mndata.load_training()
images, tlabels = mndata.load_testing()
features = features + images
features = np.array(features)
labels = labels + tlabels
labels = np.reshape(np.array(labels),(-1,1))
labels = labels == 8
ones = np.array([i for i in range(labels.shape[0]) if labels[i]==1])
np.random.shuffle(ones)
ones = np.reshape(ones,(5, -1))

#keys to feature where label is 0
zeros = np.array([i for i in range(labels.shape[0]) if labels[i]==0])
np.random.shuffle(zeros)
zeros = np.reshape(zeros,(5, -1))

ind = list(range(5))
ind = np.delete(ind, 2)
print(zeros[ind].flatten())
trainSet = np.concatenate((zeros[ind].flatten(), ones[ind].flatten())).flatten()
print(trainSet.shape)
from PIL import Image
img = Image.fromarray(np.reshape(features[trainSet[79]].astype(np.uint8),(28,28)), 'L')
img.show()
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img,volatile=True), size)).data

img = torch.from_numpy(np.reshape(features[trainSet[79]].astype(np.uint8),(28,28)))
print(img.size())
img = resize2d(img, (224,224))
print(img.size())'''

import numpy as np



original_array = np.loadtxt("test.txt")

import matplotlib.pyplot as plt
plt.plot(original_array)
plt.show()
	

