import numpy as np
import torch
import torch.nn as nn
from utils.misc import *
from utils.imagenet import *
from torchvision import models 

def load_resnet50(net, classifier, args):
    
    if args.ckpt:
        filename = args.resume + '/ckpt_epoch_{:d}.pth'.format(args.ckpt)
    else:
        filename = args.resume + '/ckpt.pth'
    ckpt = torch.load(filename)
    state_dict = ckpt['model']

    net_dict = {}
    for k, v in state_dict.items():
        k = k.replace("encoder.", "ext.")
        k = k.replace("fc.", "head.fc.")
        net_dict[k] = v

    try:
        net.load_state_dict(net_dict, strict=False)
    except RuntimeError:
        for key in ['head.fc.weight', 'head.fc.bias']:
            del net_dict[key]
        net.load_state_dict(net_dict, strict=False)

    print('Loaded model trained on classification:', filename)

def build_resnet50(args):

    print('Building ResNet50...')
    if args.dataset == 'cifar10':
        classes = 10
    elif args.dataset == "cifar100":
        classes = 100
    elif (args.dataset == "visda") or ("imagenet" in args.dataset):
        classes = 1000

    if args.dataset == 'imagenet-c':
        from torchvision.models import resnet50, ResNet50_Weights
        from models.BigResNet import LinearClassifier,Net

        classifier = LinearClassifier(num_classes=classes).cuda()
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        state_dict = model.load_state_dict()
        classifier_dict = {}
        for k, v in state_dict.items():
            if k.startswith('fc.'):
                classifier_dict[k] = v
        classifier.load_state_dict(classifier_dict)
        del(model.fc)
        ext = model
        net = Net(ext, classifier).cuda()

    else:
        from models.BigResNet import resnet50, LinearClassifier, Net
        classifier = LinearClassifier(num_classes=classes).cuda()
        ext = resnet50()
        net = Net(ext, classifier).cuda()

    return net, ext, classifier

def test(dataloader, model, dataset, sslabel=None):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    model.eval()
    correct = []
    losses = []
    by_cls = False
    if dataset == 'visda':
        by_cls, num_classes = True, 12
    if by_cls:
        correct_cls, labels_ls = [], []
    for _, ((inputs, _), labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            if dataset == 'imagenet-r':
                outputs = model(inputs)[:,imagenet_r_mask]
            elif dataset == 'imagenet-a':
                outputs = model(inputs)[:, indices_in_1k]
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.cpu())
            _, predicted = outputs.max(1)
            correct.append(predicted.eq(labels).cpu())
            if by_cls:
                labels_ls.append(labels.cpu())
                correct_ea = torch.stack([torch.tensor(sum(labels.cpu()[labels.cpu()==i]==predicted.cpu()[labels.cpu()==i])) for i in range(num_classes)])
                correct_cls.append(correct_ea)
    correct = torch.cat(correct).numpy()
    losses = torch.cat(losses).numpy()
    
    if by_cls:
        labels_ls = torch.cat(labels_ls).numpy()
        correct_cls = torch.div(torch.vstack(correct_cls).float().sum(0), torch.tensor([sum(labels_ls==i) for i in range(num_classes)]))
        
    model.train()
    if not by_cls:
        return 1-correct.mean(), correct, losses
    else:
        return 1-correct.mean(), correct, losses, 1-correct_cls
