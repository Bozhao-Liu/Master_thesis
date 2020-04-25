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
parser.add_argument('--lrDecay', default=0.8, type=float,
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

		loss_track = loss_track + train(args, dataloaders['train'], model, loss_fn, optimizer, epoch) #keep track of training loss

		# evaluate on validation set
		val_loss, AUC = get_AUC(validate(dataloaders['val'], model, loss_fn))
		logging.warning('    Loss {loss:.4f};  AUC {AUC:.4f}\n'.format(loss=val_loss, AUC=AUC))

		# remember best loss and save checkpoint		
		if best_AUC < AUC:
			logging.warning('    Saving Best AUC model\n')
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_AUC': AUC,
				'optimizer' : optimizer.state_dict(),
				}, args, CViter)
		best_AUC = max(best_AUC, AUC)
		learning_rate_decay(optimizer, args.lrDecay)
	gc.collect()
	del optimizer
	return loss_track

def learning_rate_decay(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
	

def train(args, train_loader, model, loss_fn, optimizer, epoch):
	losses = AverageMeter()

	# switch to train mode
	model.train()
	loss = []
	now = datetime.now()
	with tqdm(total=len(train_loader)) as t:
		for i, (datas, label) in enumerate(train_loader):
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
			losses.update(cost.cpu().data.numpy(), len(datas))
			if i%2 == 0:
				loss.append(losses())

			# compute gradient and do SGD step
			logging.info("        Compute gradient and do SGD step")
			optimizer.zero_grad()
			cost.backward()
			optimizer.step()
		
			gc.collect()
			t.set_postfix(loss='{:05.3f}'.format(losses()))
			t.update()

	save_TimeTrack_to_ini(args, datetime.now() - now, len(train_loader)) #save time used to evalutate loss function#
	return loss
	

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
	loss_fn = get_loss(args.loss)
	if args.lrDecay != 1.0:
		args.loss = args.loss + '_{}LrD_'.format(str(args.lrDecay))
	netlist = model_loader.get_model_list(args.network)
	for network in netlist:
		plt.clf()
		args.network = network
		set_logger(args.model_dir, args.network, args.log)

		params = set_params(args.model_dir, network)
		
		model = model_loader.loadModel(params, netname = args.network, dropout_rate = params.dropout_rate)
		model.cuda()

		cudnn.benchmark = True

		for Testiter in range(params.CV_iters):
			for CViter in range(params.CV_iters-1):
				if CViter !=0 or Testiter !=0:
					model.apply(model_loader.weight_ini)
					logging.warning('Cross Validation on iteration {}/{}, Nested CV on {}/{}'.format(Testiter + 1, params.CV_iters, CViter + 1, params.CV_iters -1))
					
					if args.train:
						save_loss_log(args, (Testiter,CViter), 
								train_model(args, params, loss_fn, model, (Testiter,CViter), network))
					evalmatices[network].append(save_ROC(	args, 
									params.CV_iters, 
									outputs = validate(	fetch_dataloader([], params), 
												resume_model(args, model, (Testiter,CViter)), 
												loss_fn)[1]))
					get_next_CV_set(params.CV_iters)

		#add the AUC SD to the current model result
		add_AUC_to_ROC(args, params.CV_iters, evalmatices[network])	
		plot_learningCurve(args, params.CV_iters)
		Store_AUC_to_ini(args, evalmatices[network])
	
	plot_AUC_SD(args.loss, evalmatices, netlist, args.lrDecay)

		
if __name__ == '__main__':
	main()
