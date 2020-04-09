import os

from mnist.loader import MNIST
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from itertools import chain
import torchvision.transforms as transforms


class DatasetWrapper:
	class __DatasetWrapper:
		"""
		A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
		"""
		def __init__(self, cv_iters):
			"""
			create df for features and labels
			remove samples that are not shared between the two tables
			"""
			assert cv_iters > 2, 'Cross validation folds must be more than 2 folds'
			self.cv_iters = cv_iters
			mndata = MNIST('data')
			self.features, self.labels = mndata.load_training()
			images, labels = mndata.load_testing()
			self.features = self.features + images
			self.features = np.array(self.features)
			self.labels = self.labels + labels
			self.labels = np.reshape(np.array(self.labels),(-1,1))
			self.labels = self.labels == 8  #6825 samles of 8 in total of 70000 samples

			self.shuffle()

		def shuffle(self):
			"""
			categorize sample ID by label
			"""
			#keys to feature where label is 1
			self.ones = np.array([i for i in range(self.labels.shape[0]) if self.labels[i]==1])
			self.ones = np.reshape(self.ones,(self.cv_iters, -1))

			#keys to feature where label is 0
			self.zeros = np.array([i for i in range(self.labels.shape[0]) if self.labels[i]==0])
			self.zeros = np.reshape(self.zeros,(self.cv_iters, -1))
			
			#index of valication set
			self.CVindex = 0
			self.Testindex = 0

		def next(self):
			'''
			rotate to the next cross validation process
			'''
			next_test = False
			if self.CVindex < self.cv_iters-1:
				self.CVindex += 1
				if self.Testindex < self.cv_iters-1:
					if self.Testindex == self.CVindex:
						self.CVindex += 1
				else:
					if self.Testindex == self.CVindex:
						self.CVindex = 0
						next_test = True
			else:
				self.CVindex = 0
				next_test = True
			
			if next_test:
				if self.Testindex < self.cv_iters-1:
					self.Testindex += 1
				else:
					self.Testindex = 0


	instance = None
	def __init__(self, cv_iters, shuffle = 0):
		if not DatasetWrapper.instance:
			DatasetWrapper.instance = DatasetWrapper.__DatasetWrapper(cv_iters)

		if shuffle:
			DatasetWrapper.instance.shuffle()

	def __getattr__(self, name):
		return getattr(self.instance, name)

	def features(self, key):
		"""
		Args: 
			key:(string) value from dataset	
		Returns:
			features in list	
		"""
		return DatasetWrapper.instance.features[key]

	def label(self, key):
		"""
		Args: 
			key:(string) the sample key/id	
		Returns:
			label to number 8 or other
		"""
		return DatasetWrapper.instance.labels[key]

	def next(self):
		DatasetWrapper.instance.next()

	def shuffle(self):
		DatasetWrapper.instance.shuffle()

	def __trainSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of trainning set
		"""

		ind = list(range(DatasetWrapper.instance.cv_iters))
		ind = np.delete(ind, [DatasetWrapper.instance.CVindex, DatasetWrapper.instance.Testindex])

		trainSet = np.concatenate((DatasetWrapper.instance.zeros[ind].flatten(), DatasetWrapper.instance.ones[ind].flatten())).flatten()
		np.random.shuffle(trainSet)
		return trainSet
	
	def __valSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of validation set
		"""

		valSet = np.concatenate((DatasetWrapper.instance.zeros[DatasetWrapper.instance.CVindex].flatten(), DatasetWrapper.instance.ones[DatasetWrapper.instance.CVindex].flatten())).flatten()
		np.random.shuffle(valSet)
		return valSet

	def __testSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of full dataset
		"""

		testSet = np.concatenate((DatasetWrapper.instance.zeros[DatasetWrapper.instance.Testindex].flatten(), DatasetWrapper.instance.ones[DatasetWrapper.instance.Testindex].flatten())).flatten()
		np.random.shuffle(testSet)
		return testSet

	def getDataSet(self, dataSetType = 'train'):
		"""
		Args: 
			dataSetType: (string) 'train' or 'val'	
		Returns:
			dataset: (np.ndarray) array of key/id of data set
		"""

		if dataSetType == 'train':
			return self.__trainSet()

		if dataSetType == 'val':
			return self.__valSet()

		if dataSetType == 'test':
			return self.__testSet()

		return self.__testSet()
		


class imageDataset(Dataset):
	"""
	A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
	"""
	def __init__(self, dataSetType, CV_iters):
		"""
		initialize DatasetWrapper
		"""
		self.DatasetWrapper = DatasetWrapper(CV_iters)

		self.samples = self.DatasetWrapper.getDataSet(dataSetType)

		self.transformer = [
				transforms.Compose([
					transforms.Grayscale(num_output_channels=1), # convert RGB image to greyscale (optional, 1 vs. 3 channels)
					transforms.RandomCrop(size = 28, padding = 2),  # randomly Crop image 
					transforms.RandomRotation(10, fill=(0,)), # randomly rotate image by 10 degrees
					transforms.ToTensor()]),  # transform it into a torch tensor
				transforms.Compose([
					transforms.Grayscale(num_output_channels=1), # convert RGB image to greyscale (optional, 1 vs. 3 channels)
					transforms.ToTensor()])]

	def __len__(self):
		# return size of dataset
		return len(self.samples)



	def __getitem__(self, idx):
		"""
		Fetch feature and labels from dataset using index of the sample.

		Args:
		    idx: (int) index of the sample

		Returns:
		    feature: (Tensor) feature image
		    label: (int) corresponding label of sample
		"""
		sample = self.samples[idx]
		from PIL import Image
		image = Image.fromarray(np.reshape(self.DatasetWrapper.features(sample).astype(np.uint8), (28,28)), 'L')
		label = self.DatasetWrapper.label(sample)
		image = self.transformer[int(label)](image)
		return image, label


def fetch_dataloader(types, params):
	"""
	Fetches the DataLoader object for each type in types.

	Args:
	types: (list) has one or more of 'train', 'val'depending on which data is required '' to get the full dataSet
	params: (Params) hyperparameters

	Returns:
	data: (dict) contains the DataLoader object for each type in types
	"""
	dataloaders = {}
	
	if len(types)>0:
		for split in types:
			if split in ['train', 'val', 'test']:
				dl = DataLoader(imageDataset(split, params.CV_iters), batch_size=params.batch_size, shuffle=True,
					num_workers=params.num_workers,
					pin_memory=params.cuda)

				dataloaders[split] = dl
	else:
		dl = DataLoader(imageDataset('',params.CV_iters), batch_size=params.batch_size, shuffle=True,
			num_workers=params.num_workers,
			pin_memory=params.cuda)

		return dl

	return dataloaders

def get_next_CV_set(CV_iters):
	Wrapper = DatasetWrapper(CV_iters)
	Wrapper.next()
