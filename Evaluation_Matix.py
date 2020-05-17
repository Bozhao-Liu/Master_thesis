import os
import torch
import logging
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_loss_log
epslon = 1e-8

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
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],torch.log(outputs[:, i]+epslon)), weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i]+epslon)))) for i in range(outputs.shape[1])]
	return -torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def L1_loss(outputs, labels):
	return np.sum(labels*(1-outputs)+(1 - labels)*outputs)

def Exp_UEW_BCE_loss(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],1.0/(outputs[:, i]+epslon) - 1), weights[1]*torch.mul(1 - labels[:, i],1.0/(1 - outputs[:, i]+epslon)-1))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Exp_UEW_BCE_loss_BCE_balanced(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],1.0/(outputs[:, i]+epslon) - 1), -weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i]+epslon)))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Exp_UEW_BCE_loss_focal_balanced(outputs, labels, gamma = 2):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(	torch.mul(labels[:, i],1.0/(outputs[:, i]+epslon) - 1), 
					-torch.mul(torch.pow(outputs[:, i], gamma), torch.mul(1 - labels[:, i], torch.log(1 - outputs[:, i]+epslon))))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Focal_loss(outputs, labels, gamma = (2, 2)):
	loss = [torch.sum(torch.add(	torch.mul(torch.pow(1 - outputs[:, i], gamma[0]), torch.mul(labels[:, i], torch.log(outputs[:, i]+epslon))), 
					torch.mul(torch.pow(outputs[:, i], gamma[1]), torch.mul(1 - labels[:, i], torch.log(1 - outputs[:, i]+epslon))))) for i in range(outputs.shape[1])]
	return -torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def get_loss(loss_name):
	loss_name = loss_name.lower()
	if loss_name == 'bce':
		return UnevenWeightBCE_loss
	elif loss_name == 'exp_bce': 
        	return Exp_UEW_BCE_loss
	elif loss_name == 'focal': 
        	return Focal_loss
	elif loss_name == 'expbce_focal_balanced': 
        	return Exp_UEW_BCE_loss_focal_balanced
	elif loss_name == 'expbce_bce_balanced': 
        	return Exp_UEW_BCE_loss_BCE_balanced
	else:
		logging.warning("No loss function with the name {} found, please check your spelling.".format(loss_name))
		logging.warning("loss function List:")
		logging.warning("    BCE")
		logging.warning("    EXP_BCE")
		logging.warning("    focal")
		logging.warning("    EXPBCE_Focal_Balanced")
		logging.warning("    EXPBCE_BCE_Balanced")
		import sys
		sys.exit()

def get_loss_list():
	return ['BCE','EXP_BCE','focal','EXPBCE_Focal_Balanced','EXPBCE_BCE_Balanced']

def get_loss_color():
	return {'BCE': 'purple', 'EXP_BCE': 'yellow','focal': 'lime','EXPBCE_Focal_Balanced': 'b', 'EXPBCE_BCE_Balanced': 'r'}

def get_loss_label(loss):
	dic = {'BCE': 'BCE', 'EXP_BCE': 'EXP_BCE', 'focal': 'focal', 'EXPBCE_Focal_Balanced': 'F-EXPBCE', 'EXPBCE_BCE_Balanced': 'B-EXPBCE'}
	return dic[loss]

def save_ROC(args, cv_iter, outputs):
	ROC_png_file = os.path.join(args.model_dir, args.network)
	ROC_png_file = os.path.join(ROC_png_file, '{network}_{loss}_CV{cv_iter}_ROC.PNG'.format(network = args.network, loss = args.loss, cv_iter = cv_iter))
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

def get_eval_matrix(outputs):
	AUC = 0
	TP_rates = []
	FP_rates = []
	FP_rate_pre = 1
	TP_rate_pre = 1
	best_F1 = 0
	best_cutoff = 0
	fpr, tpr, thresholds = metrics.roc_curve(outputs[1], outputs[0], pos_label=1)
	AUC = metrics.auc(fpr, tpr)
	F1 = [metrics.f1_score(outputs[1], outputs[0] > cutoff) for cutoff in thresholds]
	best_F1 = max(F1)
	best_cutoff = thresholds[F1.index(best_F1)]
		
	return [AUC, best_F1, best_cutoff]

def get_TP_FP(outputs):
	AUC = 0
	TP_rates = []
	FP_rates = []
	FP_rate_pre = 1
	TP_rate_pre = 1
	best_F1 = 0
	best_cutoff = 0
	fpr, tpr, thresholds = metrics.roc_curve(outputs[1], outputs[0], pos_label=1)
	mean_fpr = np.linspace(0, 1, 101)
	interp_tpr = np.interp(x = mean_fpr, xp = fpr, fp = tpr)
		
	return interp_tpr


def plot_ROC_SD(args, TP_FP, loss_list):
	color = get_loss_color()
	mean_fpr = np.linspace(0, 1, 101)
	tprs = []
	aucs = []
	fig, ax = plt.subplots()
	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
	for loss in loss_list:
		mean_tpr = np.mean(TP_FP[loss], axis=0)
		AUC = metrics.auc(mean_fpr, mean_tpr)
		mean_tpr[0] = 0
		mean_tpr[-1] = 1 
		ax.plot(mean_fpr, mean_tpr, color = color[loss], label=r'{} mean ROC ({} mean AUC)'.format(get_loss_label(loss), AUC), lw=2)
		std_tpr = np.std(TP_FP[loss], axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color = color[loss], alpha=.2)

	ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='{} ROC on Test'.format(args.network))
	ax.legend(loc="lower right")	
	ax.set_xlabel('FP_rate')
	ax.set_ylabel('TP_rate')
	ROC_png_file = '{network}_ROC.PNG'.format(network = args.network)
	plt.savefig(ROC_png_file)

def add_AUC_to_ROC(args, cv_iter, evalmatices):
	ROC_png_file = os.path.join(args.model_dir, args.network)
	ROC_png_file = os.path.join(ROC_png_file, '{network}_{loss}_CV{cv_iter}_ROC.PNG'.format(network = args.network, loss = args.loss, cv_iter = cv_iter))
	logging.warning('    adding AUC to ROC {}\n'.format(ROC_png_file))
	plt.boxplot([np.array(evalmatices).T[0]], showfliers=False)
	plt.annotate('Average F1 {:.4f}'.format(np.mean(np.array(evalmatices).T[1])),(0.9, 0.1))
	plt.annotate('Average Cutoff {:.4f}'.format(np.mean(np.array(evalmatices).T[2])),(0.9, 0.2))
	plt.savefig(ROC_png_file)

def plot_learningCurve(args, CV_iters):
	plt.clf()
	import math
	from scipy import stats
	losslist = get_loss_list()
	dev = 5
	learningCurveFile = '{network}-CV{cv_iter}-LearningCurve{dev}.PNG'.format(network = args.network, cv_iter = CV_iters, dev = dev)
	color = get_loss_color()
	fig, ax = plt.subplots()
	
	
	for loss in losslist:
		loss_logs = []
		args.loss = loss
		min_len = math.inf
		for Testiter in range(CV_iters):
			for CViter in range(CV_iters-1):
				if Testiter>1:#Testiter!=0 or CViter!=0:
					loss_log = load_loss_log(args, (Testiter,CViter))
					min_len = min(min_len, loss_log.shape[0])	
					mean_Epoch = np.linspace(0, min_len, 1001)
					loss_log = np.interp(x = mean_Epoch, xp = range(loss_log.shape[0]), fp = loss_log)				
					loss_logs.append(loss_log)
		#for i in range(len(loss_logs)):
			#loss_logs[i] = loss_logs[i][0:min_len]
	
		mean_loss = stats.trim_mean(loss_logs, 0.2, axis=0)
		ax.plot(mean_Epoch[:int(1001/dev)], mean_loss[:int(1001/dev)], color = color[loss], label=r'{} mean learning curve'.format(get_loss_label(loss)), lw=2, alpha=.8)
		std_loss = np.std(loss_logs, axis=0)
		tprs_upper = np.minimum(mean_loss + std_loss, 1)[:int(1001/dev)]
		tprs_lower = np.maximum(mean_loss - std_loss, 0)[:int(1001/dev)]
		ax.fill_between(mean_Epoch[:int(1001/dev)], tprs_lower, tprs_upper, color = color[loss], alpha=.2)
			
	ax.set_ylabel('AUC', fontsize = 'x-large')
	ax.set_xlabel('Epochs', fontsize = 'x-large')
	ax.legend(loc="lower right", fontsize = 'x-large')
	ax.set(xlim=[-0.2, min_len/dev-1])
	ax.set_title('{} Learning Curve, {} fold 1/{}'.format(args.network, CV_iters, dev), fontsize = 'x-large')
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

def plot_AUC_SD(loss, evalmatices, netlist, lrd):
	logging.warning('    Creating standard diviation image for {}'.format('-'.join(netlist)))
	AUC_png_file = 'Crossvalidation_Analysis_{}_{}_{}LrD.PNG'.format(loss, '_'.join(netlist), str(lrd))

	if len(netlist) == 0:
		return

	x = np.array(range(len(netlist)))+1
	plt.clf()
	fig, ax = plt.subplots(3)
	fig.suptitle('AUC, F1, Cutoff standard diviation')
	
	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[0])

	ax[0].boxplot(data, showfliers=False)
	ax[0].set_ylabel('AUC')
	#ax[0].set_xlabel('Network name')
	ax[0].set_xticklabels(['BCE', 'EXP_BCE', 'focal', 'F-EXPBCE', 'B-EXPBCE'], fontsize=10)

	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[1])

	ax[1].boxplot(data, showfliers=False)
	ax[1].set_ylabel('F1')
	#ax[1].set_xlabel('Network name')
	ax[1].set_xticklabels(['BCE', 'EXP_BCE', 'focal', 'F-EXPBCE', 'B-EXPBCE'], fontsize=10)

	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[2])

	ax[2].boxplot(data, showfliers=False)
	ax[2].set_ylabel('Cutoff')
	#ax[2].set_xlabel('Network name')
	ax[2].set_xticklabels(['BCE', 'EXP_BCE', 'focal', 'F-EXPBCE', 'B-EXPBCE'], fontsize=10)

	plt.xlabel('Network name')
	logging.warning('    Saving standard diviation image for {} \n'.format('-'.join(netlist)))
	plt.savefig(AUC_png_file)



