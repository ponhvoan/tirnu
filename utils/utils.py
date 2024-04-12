import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
import numpy as np
import torchvision.transforms.v2 as transforms

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def get_Cent(input_, ref):
    bs = input_.size(0)
    Cent = -ref * torch.log(input_ + 1e-5)
    Cent = torch.sum(Cent, dim=1)
    return Cent

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

####### -------------------- #######
######### Helper Functions #########
def normalize(dataset):

    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == 'visda':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'imagenet' in dataset:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError

    normalize = transforms.Normalize(mean=mean, std=std)
    te_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    return te_transforms

def simclr_transforms(dataset):
    if 'cifar' in dataset:
        size = 32
    else:
        size = 224
    return  transforms.Compose([
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize(dataset)
            ])

# ----------------------------------