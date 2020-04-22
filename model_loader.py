import os
import sys
import torch
import logging
import torch.nn as nn
def loadModel(params, netname = 'basemodel', dropout_rate = 0.5, channels = 1):
    Netpath = 'Model'
    
    Netfile = os.path.join(Netpath, netname)
    Netfile = os.path.join(Netfile, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    netname = netname.lower()
    if netname == 'basemodel':
        return loadBaseModel(channels, dropout_rate)
    if netname == 'alexnet': 
        return loadAlexnet(pretrained = False)
    elif netname == 'densenet': 
        return loadDensenet(pretrained = False, params = params)
    elif netname == 'smallresnet': 
        return loadSmallResNet()
    else:
        logging.warning("No model with the name {} found, please check your spelling.".format(netname))
        logging.warning("Net List:")
        logging.warning("    basemodel")
        logging.warning("    AlexNet")
        logging.warning("    DenseNet")
        logging.warning("    SmallResNet")
        sys.exit()

def get_model_list(netname = ''):
    netname = netname.lower()
    if netname == '':
        return ['basemodel', 'alexnet', 'densenet', 'smallresnet']

    if netname in ['basemodel', 'alexnet', 'densenet', 'smallresnet']:
        return [netname]

    logging.warning("No model with the name {} found, please check your spelling.".format(netname))
    logging.warning("Net List:")
    logging.warning("    basemodel")
    logging.warning("    AlexNet")
    logging.warning("    DenseNet")
    logging.warning("    SmallResNet")
    sys.exit()
    
def loadBaseModel(channels, dropout_rate):
    from Model.basemodel.basemodel import Base_Model
    logging.warning("Loading Base Model")
    return Base_Model(channels, dropout_rate)

def loadAlexnet(pretrained):
    from Model.alexnet.alexnet import alexnet
    print("Loading AlexNet")
    return alexnet(pretrained = pretrained, num_classes = 1)
    
def loadDensenet(pretrained, params):
    from Model.densenet.densenet import net
    print("Loading DenseNet")
    return net(str(params.version), pretrained, num_classes = 1)

def loadSmallResNet():
    from Model.smallresnet.smallresnet import smallresnet
    print("Loading SmallResNet")
    return smallresnet(num_classes=1)

def weight_ini(m):
    torch.manual_seed(230)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
    

