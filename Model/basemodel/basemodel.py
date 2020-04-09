import torch.nn as nn


class Base_Model(nn.Module):
	def __init__(self, channels, dropout_rate):
		"""
		Args:
		    params: (Params) contains num_channels
		"""
		super(Base_Model, self).__init__()
		self.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(28*28, channels), nn.Sigmoid())

	def forward(self, x):
		"""
		This function defines how we use the components of our network to operate on an input batch.

		Args:
		    X: (Variable) features.

		Returns:
		    out: (Variable) dimension batch_size x 1 with the log probabilities for the prediction.

		Note: the dimensions after each step are provided
		"""
		x = x.flatten(1)
		
		return self.fc(x)

