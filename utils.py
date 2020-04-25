import json
import logging
import os
import shutil
import torch
import numpy as np
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, model_dir, network):
        json_path = os.path.join(model_dir, network)
        json_file = os.path.join(json_path, 'params.json')
        logging.info("Loading json file {}".format(json_file))
        assert os.path.isfile(json_file), "Can not find File {}".format(json_file)
        with open(json_file) as f:
            params = json.load(f)

            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
        
    
def set_logger(model_dir, network, level = 'info'):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    log_path = os.path.join(model_dir, network)
    assert os.path.isdir(log_path), "Can not find Path {}".format(log_path)
    log_path = os.path.join(log_path, 'train.log')
    print('Saving {} log to {}'.format(level, log_path))
    level = level.lower()
    logger = logging.getLogger()
    if level == 'warning':
        level = logging.WARNING
    elif level == 'debug':
        level = logging.DEBUG
    elif level == 'error':
        level = logging.ERROR
    elif level == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    logger.setLevel(level)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def set_params(model_dir, network):
	params = Params(model_dir, network)

	# use GPU if available
	params.cuda = torch.cuda.is_available()

	# Set the random seed for reproducible experiments
	torch.manual_seed(230)
	if params.cuda: 
		torch.cuda.manual_seed(230)

	return params

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, args, CViter):
	checkpointfile = os.path.join(args.model_dir, args.network)
	checkpointfile = os.path.join(checkpointfile, 'Checkpoints')
	if not os.path.isdir(checkpointfile):
		os.mkdir(checkpointfile)
	checkpointfile = os.path.join(checkpointfile, '{network}_{loss}_{cv_iter}.pth.tar'.format(network = args.network, loss = args.loss, cv_iter = '_'.join(tuple(map(str, CViter)))))
	torch.save(state, checkpointfile)


def resume_checkpoint(args, model, optimizer, CViter):
	checkpointfile = os.path.join(args.model_dir, args.network)
	checkpointfile = os.path.join(checkpointfile, 'Checkpoints')
	checkpointfile = os.path.join(checkpointfile, '{network}_{loss}_{cv_iter}.pth.tar'.format(network = args.network, loss = args.loss, cv_iter = '_'.join(tuple(map(str, CViter)))))
	assert os.path.isfile(checkpointfile), "=> no checkpoint found at '{}'".format(checkpointfile)

	logging.info("Loading checkpoint {}".format(checkpointfile))
	checkpoint = torch.load(checkpointfile)
	start_epoch = checkpoint['epoch']
	best_AUC = checkpoint['best_AUC']
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	return start_epoch, best_AUC, model, optimizer

def resume_model(args, model, CViter):
	checkpointfile = os.path.join(args.model_dir, args.network)
	checkpointfile = os.path.join(checkpointfile, 'Checkpoints')
	checkpointfile = os.path.join(checkpointfile, '{network}_{loss}_{cv_iter}.pth.tar'.format(network = args.network, loss = args.loss, cv_iter = '_'.join(tuple(map(str, CViter)))))
	assert os.path.isfile(checkpointfile), "=> no checkpoint found at '{}'".format(checkpointfile)

	logging.info("Loading model {}".format(checkpointfile))
	model.load_state_dict(torch.load(checkpointfile)['state_dict'])
	return model

def save_TimeTrack_to_ini(args, timeused, iters):
	from datetime import datetime
	import pandas as pd

	time_location = args.loss + '.time'
	iter_location = args.loss + '.iterations'
	config = ConfigParser()
	config_name = os.path.join(args.model_dir, 'TimeTrackCompare.ini')
	if os.path.isfile(config_name):
		config.read(config_name)
		if config.has_section(args.network):
			config.read(args.network)
		else:
			config.add_section(args.network)
	else:
		config['DEFAULT'] = {time_location: '0', iter_location: '0'}
		config.add_section(args.network)

	if not time_location in config['DEFAULT']:
		config['DEFAULT'][time_location] = '0'

	if not iter_location in config['DEFAULT']:
		config['DEFAULT'][iter_location] = '0'

	section = config[args.network]

	section.get(time_location)
	timelog = pd.to_timedelta(section[time_location])+timeused
	config.set(args.network, time_location, str(timelog))


	section.get(iter_location)
	iters = iters + int(section[iter_location])
	config.set(args.network, iter_location, str(iters))

	config.set(args.network, args.loss+ '.time_per_minibatch', str(timelog/iters))

	config.write(open(config_name, 'w+'))

def Store_AUC_to_ini(args, evalmatices):
	import numpy as np

	AUC_location = args.loss + '.Average_AUC'
	F1_location = args.loss + '.Average_F1'
	CutOff_location = args.loss + '.Average_CutOff'

	config = ConfigParser()
	config_name = os.path.join(args.model_dir, 'EvalMatrixCompare.ini')
	if os.path.isfile(config_name):
		config.read(config_name)
		if config.has_section(args.network):
			config.read(args.network)
		else:
			config.add_section(args.network)
	else:
		config.add_section(args.network)

	config.set(args.network, AUC_location, str(np.mean(np.array(evalmatices).T[0])))
	config.set(args.network, F1_location, str(np.mean(np.array(evalmatices).T[1])))
	config.set(args.network, CutOff_location, str(np.mean(np.array(evalmatices).T[2])))

	config.write(open(config_name, 'w+'))
	return 0

def save_loss_log(args, CViter, data):
	loss_log = os.path.join(args.model_dir, args.network)
	loss_log = os.path.join(loss_log, 'LossLog')
	if not os.path.isdir(loss_log):
		os.mkdir(loss_log)
	loss_log = os.path.join(loss_log, '{network}_{loss}_{cv_iter}.txt'.format(network = args.network, loss = args.loss, cv_iter = '_'.join(tuple(map(str, CViter)))))
	if os.path.isfile(loss_log) and not args.resume:
		os.remove(loss_log)
	with open(loss_log, "w") as log_file:
    		np.savetxt(log_file, data)

def load_loss_log(args, CViter):
	loss_log = os.path.join(args.model_dir, args.network)
	loss_log = os.path.join(loss_log, 'LossLog')
	assert os.path.isdir(loss_log), 'No dirctory with name {}'.format(loss_log)

	loss_log = os.path.join(loss_log, '{network}_{loss}_{cv_iter}.txt'.format(network = args.network, loss = args.loss, cv_iter = '_'.join(tuple(map(str, CViter)))))
	assert os.path.isfile(loss_log), 'Cannot find Loss Log file {}'.format(loss_log)
	return np.loadtxt(loss_log)
