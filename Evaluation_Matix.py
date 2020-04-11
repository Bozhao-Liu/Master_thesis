import os
import torch
import logging
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __call__(self):
        return self.avg

def UnevenWeightBCE_loss(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],torch.log(outputs[:, i])), weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i])))) for i in range(outputs.shape[1])]
	return torch.neg(torch.stack(loss, dim=0).sum(dim=0).sum(dim=0))

def Exp_UEW_BCE_loss(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],torch.pow(outputs[:, i], - 1) - 1), -weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i])))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Focal_loss(outputs, labels, gamma = (2, 2)):
	loss = [torch.sum(torch.add(	torch.mul(torch.pow(1 - outputs[:, i], gamma[0]), torch.mul(labels[:, i], torch.log(outputs[:, i]))), 
					torch.mul(torch.pow(outputs[:, i], gamma[1]), torch.mul(1 - labels[:, i], torch.log(1 - outputs[:, i]))))) for i in range(outputs.shape[1])]
	return -torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def get_loss(loss_name):
	loss_name = loss_name.lower()
	if loss_name == 'bce':
		return UnevenWeightBCE_loss
	if loss_name == 'exp_bce': 
        	return Exp_UEW_BCE_loss
	elif loss_name == 'focal': 
        	return Focal_loss
	else:
		logging.warning("No loss function with the name {} found, please check your spelling.".format(loss_name))
		logging.warning("loss function List:")
		logging.warning("    BCE")
		logging.warning("    EXP_BCE")
		logging.warning("    Focal")
		sys.exit()

def save_ROC(args, cv_iter, outputs):
	ROC_png_file = '{network}_{loss}_CV{cv_iter}_ROC.PNG'.format(network = args.network, loss = args.loss, cv_iter = cv_iter)
	AUC = 0
	TP_rates = []
	FP_rates = []
	FP_rate_pre = 1
	TP_rate_pre = 1
	best_F1 = 0
	best_cutoff = 0
	logging.warning('Creating ROC image for {} \n'.format(args.network))
	for i in tqdm(np.linspace(0, 1, 51)):
		results = outputs[0]>i
		TP = np.sum((results+outputs[1])==2, dtype = float)
		FN = np.sum(results<outputs[1], dtype = float)
		TP_rate = TP/(TP + FN + 1e-10)  # recall
		TP_rates.append(TP_rate)
		FP = np.sum(results>outputs[1], dtype = float)
		TN = np.sum((results+outputs[1])==0, dtype = float)
		precision = TP/(TP + FP + 1e-10)
		F1 = 2*precision*TP_rate/(precision+TP_rate + 1e-10)
		if F1 > best_F1:
			best_cutoff = i
		best_F1 = max(best_F1, F1)

		FP_rate = FP/(FP + TN + 1e-10)
		FP_rates.append(FP_rate)
		AUC += (TP_rate_pre+TP_rate)*(FP_rate_pre-FP_rate)/2.0
		FP_rate_pre = FP_rate
		TP_rate_pre = TP_rate
		
	plt.plot(FP_rates,TP_rates)
	
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.title('{}_{} ROC on Test, {} fold '.format(args.network, args.loss, cv_iter))
	logging.warning('    Saving ROC plot to {}\n'.format(ROC_png_file))
	plt.savefig(ROC_png_file)
	return [AUC, best_F1, best_cutoff]

def add_AUC_to_ROC(args, cv_iter, evalmatices):
	ROC_png_file = '{network}_{loss}_CV{cv_iter}_ROC.PNG'.format(network = args.network, loss = args.loss, cv_iter = cv_iter)
	logging.warning('    adding AUC to ROC {}\n'.format(ROC_png_file))
	plt.boxplot([np.array(evalmatices).T[0]], showfliers=False)
	plt.annotate('Average F1 {:.4f}'.format(np.mean(np.array(evalmatices).T[1])),(0.9, 0.1))
	plt.annotate('Average Cutoff {:.4f}'.format(np.mean(np.array(evalmatices).T[2])),(0.9, 0.2))
	plt.savefig(ROC_png_file)

def plot_learningCurve(args, cv_iter, data):
	plt.clf()
	learningCurveFile = '{network}_{loss}_CV{cv_iter}_LearningCurve.PNG'.format(network = args.network, loss = args.loss, cv_iter = cv_iter)

	for i in range(len(data)):
		plt.plot(data[i])
	plt.ylabel('loss')
	plt.xlabel('Epochs')
	plt.title('{}_{} Learning Curve, {} fold'.format(args.network, args.loss, cv_iter))
	logging.warning('    Saving Learning Curve to {}\n'.format(learningCurveFile))
	plt.savefig(learningCurveFile)

def get_AUC(output):
	outputs = output[1] #outputs[0] as predicted probs, outputs[1] as labels
	AUC = 0
	FP_rate_pre = 1
	TP_rate_pre = 1
	for i in np.linspace(0,1):
		results = outputs[0]>i
		TP = np.sum((results+outputs[1])==2, dtype = float)
		FN = np.sum(results<outputs[1], dtype = float)
		TP_rate = TP/(TP + FN)
		FP = np.sum(results>outputs[1], dtype = float)
		TN = np.sum((results+outputs[1])==0, dtype = float)
		FP_rate = FP/(FP + TN)
		AUC += (TP_rate_pre+TP_rate)*(FP_rate_pre-FP_rate)/2.0
		FP_rate_pre = FP_rate
		TP_rate_pre = TP_rate
	return output[0], AUC

def plot_AUD_SD(loss, evalmatices, netlist):
	logging.warning('    Creating standard diviation image for {}'.format('-'.join(netlist)))
	AUC_png_file = 'Crossvalidation_Analysis_{}_{}.PNG'.format(loss, '_'.join(netlist))

	if len(netlist) == 0:
		return

	x = np.array(range(len(netlist)))+1
	plt.clf()
	fig, ax = plt.subplots(3)
	fig.suptitle('AUC, F1, Cutoff standard diviation for {}'.format('-'.join(netlist)))
	
	data = []
	for i in range(len(netlist)):
		data.append(np.array(evalmatices[netlist[i]]).T[0])

	ax[0].boxplot(data, showfliers=False)
	ax[0].set_ylabel('AUC')
	#ax[0].set_xlabel('Network name')
	ax[0].set_xticklabels(netlist, fontsize=10)

	data = []
	for i in range(len(netlist)):
		data.append(np.array(evalmatices[netlist[i]]).T[1])

	ax[1].boxplot(data, showfliers=False)
	ax[1].set_ylabel('F1')
	#ax[1].set_xlabel('Network name')
	ax[1].set_xticklabels(netlist, fontsize=10)

	data = []
	for i in range(len(netlist)):
		data.append(np.array(evalmatices[netlist[i]]).T[2])

	ax[2].boxplot(data, showfliers=False)
	ax[2].set_ylabel('Cutoff')
	#ax[2].set_xlabel('Network name')
	ax[2].set_xticklabels(netlist, fontsize=10)

	plt.xlabel('Network name')
	logging.warning('    Saving standard diviation image for {} \n'.format('-'.join(netlist)))
	plt.savefig(AUC_png_file)



