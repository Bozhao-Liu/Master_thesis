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
parser.add_argument('--lrDecay', default=1, type=float,
			help='learning rate decay rate')


		
if __name__ == '__main__':
	args = parser.parse_args()
	params = set_params(args.model_dir, args.network)
	plot_learningCurve(args, params.CV_iters)
