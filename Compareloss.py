import argparse

from numpy import isnan
import torch
import logging
import gc


from collections import defaultdict
import torch.backends.cudnn as cudnn
from Evaluation_Matix import *
from utils import *
import model_loader
from data_loader import fetch_dataloader, get_next_CV_set
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description='PyTorch Deep Neural Net Training')
parser.add_argument('--model_dir', default='Model', 
			help="Directory containing params.json")
parser.add_argument('--network', type=str, default = '',
			help='select network to train on. leave it blank means train on all model')
parser.add_argument('--log', default='warning', type=str,
			help='set logging level')
parser.add_argument('--lrDecay', default=0.8, type=float,
			help='learning rate decay rate')

	

def validate(val_loader, model, loss_fn):
	logging.info("Validating")
	logging.info("Initializing measurement")
	losses = AverageMeter()
	outputs = [np.array([]), np.array([])]

	# switch to evaluate mode
	model.eval()

	for i, (datas, label) in enumerate(val_loader):
		logging.info("    Sample {}:".format(i))
		logging.info("        Loading Varable")
		input_var = torch.autograd.Variable(datas.cuda())
		label_var = torch.autograd.Variable(label.cuda()).double()

		# compute output
		logging.info("        Compute output")
		output = model(input_var).double()
		outputs[0] = np.concatenate((outputs[0], output.cpu().data.numpy().flatten()))
		outputs[1] = np.concatenate((outputs[1], label_var.cpu().data.numpy().flatten()))
		loss = loss_fn(output, label_var)
		assert not isnan(loss.cpu().data.numpy()),  "Overshot loss, Loss = {}".format(loss.cpu().data.numpy())
		# measure record cost
		losses.update(loss.cpu().data.numpy(), len(datas))
	
	return losses(), outputs


def main():
	args = parser.parse_args()
	evalmatices = defaultdict(list)
	
	# define loss function
	loss_list = get_loss_list()
	netlist = model_loader.get_model_list(args.network)
	
	plt.clf()
	set_logger(args.model_dir, args.network, args.log)

	params = set_params(args.model_dir, args.network)
	
	model = model_loader.loadModel(params, netname = args.network, dropout_rate = params.dropout_rate)
	model.cuda()

	cudnn.benchmark = True
	for loss in loss_list:
		args.loss = loss
		loss_fn = get_loss(loss)
		if args.lrDecay != 1.0:
			args.loss = args.loss + '_{}LrD_'.format(str(args.lrDecay))
		for Testiter in range(params.CV_iters):
			for CViter in range(params.CV_iters-1):
				if CViter !=0 or Testiter !=0:
					model.apply(model_loader.weight_ini)
					logging.warning('Cross Validation on iteration {}/{}, Nested CV on {}/{}'.format(Testiter + 1, params.CV_iters, CViter + 1, params.CV_iters -1))
					
					evalmatices[loss].append(get_eval_matrix(outputs = validate(	fetch_dataloader([], params), 
												resume_model(args, model, 														(Testiter,CViter)), 
												loss_fn)[1]))
				get_next_CV_set(params.CV_iters)
	
	plot_AUC_SD(args.network, evalmatices, loss_list, args.lrDecay)

		
if __name__ == '__main__':
	main()
