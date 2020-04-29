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
parser.add_argument('--train', default = False, type=str2bool, 
			help="specify whether train the model or not (default: False)")
parser.add_argument('--model_dir', default='Model', 
			help="Directory containing params.json")
parser.add_argument('--resume', default = False, type=str2bool, 
			help='path to latest checkpoint (default: True)')
parser.add_argument('--network', type=str, default = '',
			help='select network to train on. leave it blank means train on all model')
parser.add_argument('--loss', type=str, default = 'BCE',
			help='select loss function to train with. ')
parser.add_argument('--log', default='warning', type=str,
			help='set logging level')
parser.add_argument('--lrDecay', default=1, type=float,
			help='learning rate decay rate')


def train_model(args, params, loss_fn, model, CViter, network):
	start_epoch = 0
	best_AUC = 0
	# define optimizer		
	optimizer = torch.optim.Adam(model.parameters(), params.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=params.weight_decay, amsgrad=False)

	if args.resume:
		logging.info('Resuming Check point: {}'.format(args.resume))
		start_epoch, best_AUC, model, optimizer = resume_checkpoint(args, model, optimizer, CViter)

	logging.info("fetch_dataloader")
	dataloaders = fetch_dataloader(['train', 'val'], params) 
	loss_track = []

	for epoch in range(start_epoch, start_epoch + params.epochs):
		logging.warning('CV [{}/{},{}/{}], Training Epoch: [{}/{}]'.format(CViter[1] + 1, params.CV_iters-1, CViter[0] + 1, params.CV_iters, epoch+1, start_epoch + params.epochs))

		loss_track = loss_track + train(args, dataloaders, model, loss_fn, optimizer, epoch) #keep track of training loss

		# evaluate on validation set
		loss_track.append(get_eval_matrix(validate(dataloaders['val'], model, loss_fn))[1])
		

		learning_rate_decay(optimizer, args.lrDecay)
	gc.collect()
	del optimizer
	return loss_track

def learning_rate_decay(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
	

def train(args, dataloaders, model, loss_fn, optimizer, epoch):
	# switch to train mode
	model.train()
	loss = []
	now = datetime.now()
	with tqdm(total=len(dataloaders['train'])) as t:
		for i, (datas, label) in enumerate(dataloaders['train']):
			logging.info("    Sample {}:".format(i))
			
			logging.info("        Loading Varable")
			input_var = torch.autograd.Variable(datas.cuda())
			label_var = torch.autograd.Variable(label.cuda()).double()

			# compute output
			logging.info("        Compute output")
			output = model(input_var).double()

			# measure record cost
			cost = loss_fn(output, label_var)
			assert not isnan(cost.cpu().data.numpy()),  "Gradient exploding, Loss = {}".format(cost.cpu().data.numpy())

			# compute gradient and do SGD step
			logging.info("        Compute gradient and do SGD step")
			optimizer.zero_grad()
			cost.backward()
			optimizer.step()
		
			if i%2 == 0:
				loss.append(get_eval_matrix(validate(dataloaders['val'], model, loss_fn))[1])
			gc.collect()
			t.set_postfix(loss='{:05.3f}'.format(cost.cpu().data.numpy()))
			t.update()
			del input_var
			del label_var
			del output

	return loss
	

def validate(val_loader, model, loss_fn):
	logging.info("Validating")
	logging.info("Initializing measurement")
	outputs = [np.array([]), np.array([])]

	# switch to evaluate mode


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
		del input_var
		del label_var
		del output

	return outputs

def save_f1_log(args, CViter, data):
	f1_log = os.path.join(args.model_dir, args.network)
	f1_log = os.path.join(f1_log, 'f1Log')
	if not os.path.isdir(f1_log):
		os.mkdir(f1_log)
	f1_log = os.path.join(f1_log, '{network}_{loss}_{cv_iter}.txt'.format(network = args.network, loss = args.loss, cv_iter = '_'.join(tuple(map(str, CViter)))))
	if os.path.isfile(f1_log) and not args.resume:
		os.remove(f1_log)
	with open(f1_log, "w") as log_file:
    		np.savetxt(f1_log, data)

def load_f1_log(args, CViter):
	f1_log = os.path.join(args.model_dir, args.network)
	f1_log = os.path.join(f1_log, 'f1Log')
	assert os.path.isdir(f1_log), 'No dirctory with name {}'.format(f1_log)

	f1_log = os.path.join(f1_log, '{network}_{loss}_{cv_iter}.txt'.format(network = args.network, loss = args.loss, cv_iter = '_'.join(tuple(map(str, CViter)))))
	assert os.path.isfile(f1_log), 'Cannot find Loss Log file {}'.format(f1_log)
	return np.loadtxt(f1_log)

def plot_F1learningCurve(args, CV_iters):
	learningCurveFile = os.path.join(args.model_dir, args.network)
	learningCurveFile = os.path.join(learningCurveFile, '{network}_CV{cv_iter}_F1LearningCurve.PNG'.format(network = args.network, cv_iter = 5))
	plt.plot(load_f1_log(args, CV_iters))
	plt.legend(get_loss_list())
	plt.ylabel('F1')
	plt.xlabel('Epochs')
	plt.title('{} F1Learning Curve, {} fold'.format(args.network,  5))
	logging.warning('    Saving Learning Curve to {}\n'.format(learningCurveFile))
	plt.savefig(learningCurveFile)

def main():
	args = parser.parse_args()
	evalmatices = defaultdict(list)
	# define loss function
	loss_list = get_loss_list()
	
	plt.clf()
	set_logger(args.model_dir, args.network, args.log)

	params = set_params(args.model_dir, args.network)
	
	model = model_loader.loadModel(params, netname = args.network, dropout_rate = params.dropout_rate)
	model.cuda()

	cudnn.benchmark = True
	for loss in loss_list:
		args.loss = loss
		loss_fn = get_loss(loss)

		model.apply(model_loader.weight_ini)
		logging.warning('Training {} with {}'.format(args.network, args.loss))
		
		if args.train:
			save_f1_log(args, (0,0), 
				train_model(args, params, loss_fn, model, (0,0), args.network))
		plot_F1learningCurve(args, (0,0))
		logging.warning('get_next_CV_set')

		
if __name__ == '__main__':
	main()
