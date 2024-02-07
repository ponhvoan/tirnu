from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *
from utils.corruptions import corrupt
from utils.online import FeatureQueue
# ----------------------------------

import copy
import time
import pandas as pd
from PIL import Image

import random
import numpy as np

from utils.visualization import *

from utils.utils import Entropy, get_uncert
from utils.utils import op_copy, lr_scheduler, update_statistics, configure_model, collect_params
from utils.corruptions import corrupt

from utils.meib import kernel_width, calculate_MI
import torchvision.transforms.v2 as transforms
# from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import json

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

inv_normalize_cifar10 = transforms.Normalize(
    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    std=[1/0.2023, 1/0.1994, 1/0.2010]
)

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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default='data/cifar')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--queue_size', default=128, type=int)
parser.add_argument('--aug_size', default=1, type=int)
parser.add_argument('--ks', default=10, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--workers', default=36, type=int)
parser.add_argument('--num_sample', default=100000, type=int)
parser.add_argument('--alpha', default=1.01, type=float)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--nepoch', default=5, type=int, help='maximum number of epoch for ttt')
parser.add_argument('--bnepoch', default=1, type=int, help='first few epochs to update bn stat')
parser.add_argument('--delayepoch', default=0, type=int)
parser.add_argument('--stopepoch', default=2, type=int)
########################################################################
parser.add_argument('--outf', default='results/cifar10_tirnu_resnet50')
########################################################################
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--corruption', default='snow')
parser.add_argument('--resume', default='models/source_weights/cifar10_resnet50', help='directory of pretrained model')
parser.add_argument('--ckpt', default=None, type=int)
parser.add_argument('--fix_ssh', action='store_true')
########################################################################
parser.add_argument('--method', default='tirnu', choices=['tirnu'])
########################################################################
parser.add_argument('--model', default='resnet50', help='resnet50')
parser.add_argument('--save_every', default=100, type=int)
########################################################################
parser.add_argument('--tsne', action='store_true')
########################################################################
parser.add_argument('--cls_par', type=float, default=0.001)
parser.add_argument('--ent_par', type=float, default=1.0)
parser.add_argument('--ent2_par', type=float, default=0.2)
parser.add_argument('--uncert_par', type=float, default=0.1)
parser.add_argument('--Izn_par', type=float, default=0.1)
parser.add_argument('--ent', type=bool, default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--ent2', type=bool, default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--uncert', type=bool, default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--Izn', type=bool, default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--t_stats', type=bool, default=False)
parser.add_argument('--configure', type=bool, default=False)
parser.add_argument('--transform', type=str, choices=['T1', 'T2', 'T3', 'T4', 'mixed'], default='T1')
########################################################################
parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()
# if args.Izn:
#     args.outf = f'results/{args.dataset}_Izn_joint_resnet50'

print(args)
my_makedir(args.outf)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# -------------------------------

net, ext, classifier = build_resnet50(args)

_, teloader = prepare_test_data(args)

# -------------------------------

args.batch_size = min(args.batch_size, args.num_sample)
_, trloader = prepare_test_data(args, num_sample=args.num_sample)

# -------------------------------

print('Resuming from %s...' %(args.resume))

if 'imagenet' not in args.dataset:
    load_resnet50(net, classifier, args)

if args.t_stats:
    ext = update_statistics(ext)
if args.configure:
    ext = configure_model(ext)
    params, param_names = collect_params(ext)
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    optimizer = op_copy(optimizer)
    
## Freeze classifier module
for k, v in classifier.named_parameters():
    v.requires_grad = False

# ----------- Test ------------

if args.tsne:
    args_src = copy.deepcopy(args)
    args_src.corruption = 'original'
    _, srcloader = prepare_test_data(args_src)
    feat_src, label_src, _ = visu_feat(ext, srcloader, args.dataset, os.path.join(args.outf, 'original.jpg'))

    feat_tar, label_tar, _ = visu_feat(ext, teloader, args.dataset, os.path.join(args.outf, args.corruption + '_test_class.jpg'))
    nuisance_tar, feat2_tar, nlabel_tar, _, _ = visu_nuisance(ext, teloader, args.dataset, os.path.join(args.outf, args.corruption + f'_tirnu0_{args.transform}.jpg'), args.transform)
    comp_feat(nuisance_tar, nlabel_tar, feat_tar, label_tar, os.path.join(args.outf, args.corruption + f'_nuisance_marginal0{args.transform}.jpg'))

all_err_cls = []

print('Running...')

if args.dataset == 'visda':
    print('Error (%)\t\ttest\t\ttirnu\t\tper class')
    err_cls, _, _, err_each = test(teloader, net, args.dataset)
    print(('Epoch %d/%d:' %(0, args.nepoch)).ljust(24) +
            '%.2f\t\t\t\t' %(err_cls*100) +
            ' '.join(map(str, np.round(err_each.numpy()*100, 2))))
else:
    print('Error (%)\t\ttest\t\ttirnu')
    err_cls = test(teloader, net, args.dataset)[0]
    print(('Epoch %d/%d:' %(0, args.nepoch)).ljust(24) +
            '%.2f\t\t' %(err_cls*100))

# -------------------------------

if not args.configure:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    optimizer = op_copy(optimizer)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)

# ----------- Improved Test-time Training ------------

losses = AverageMeter('Loss', ':.4e')

for epoch in range(1, args.nepoch+1):

    # Construct queue:
    if args.queue_size > args.batch_size:
        queue_ext_feat = FeatureQueue(length=args.queue_size-args.batch_size)
        queue_ext_feat2 = FeatureQueue(length=args.queue_size-args.batch_size)
        queue_ext_c = FeatureQueue(length=args.queue_size-args.batch_size)
    tic = time.time()
    ext.train()

    # optimizer = lr_scheduler(optimizer, epoch, 30)
    for batch_idx, ((inputs, x_orig), labels) in enumerate(trloader):

        optimizer.zero_grad()
        
        features_test = ext(inputs.cuda())

        # Append and pop out features in queue
        if args.queue_size > args.batch_size:
            feat_queue = queue_ext_feat.get()
            queue_ext_feat.update(features_test)
            if feat_queue is not None:
                features_test = torch.cat([features_test, feat_queue.cuda()])

        outputs_test = classifier(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            entropy_loss = torch.mean(Entropy(softmax_out))
            classifier_loss += entropy_loss

        if args.Izn:
            cfsets = []
            features_test2 = []
            if args.transform == 'mixed': 
                mix_transform = random.sample(['T1', 'T2', 'T3', 'T4'], k=1)[0]
            else:
                mix_transform = None
            for _ in range(args.aug_size):
                if args.transform == 'T1' or mix_transform == 'T1':
                    x_aug = simclr_transforms(args.dataset)(x_orig)
                    z_aug = ext(x_aug.cuda())
                elif args.transform == 'T2' or mix_transform == 'T2':
                    x_aug = apply_lp_corruption(x_orig, 8, combine_train_corruptions=True, train_corruptions=train_corruptions,
                                            concurrent_combinations=1, max=False, noise='uniform-l2', epsilon=0.25)
                    x_aug = normalize(args.dataset)(x_aug)
                    # x_aug = transforms.Normalize(mean=x_aug.mean(dim=(0,2,3)), std=x_aug.std(dim=(0,2,3)))(x_aug)
                    z_aug = ext(x_aug.cuda())
                elif args.transform == 'T3' or mix_transform == 'T3':
                    x_origT3 = torch.stack(transforms.PILToTensor()([transforms.ToPILImage()(x_orig[i]) for i in range(len(x_orig))]), dim=0)
                    x_aug = []

                    corruption = random.sample(['snow', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression'], k=1)[0]
                    for i in range(len(x_origT3)):
                        x_aug_i = transforms.ToTensor()(Image.fromarray(corrupt(x_origT3[i].permute(1,2,0).numpy(), severity=args.level, corruption_name=args.corruption)).convert('RGB'))
                        x_aug.append(x_aug_i)
                    x_aug = torch.stack(x_aug, dim=0)
                    x_aug = normalize(args.dataset)(x_aug)
                    z_aug = ext(x_aug.cuda())

                elif args.transform == 'T4' or mix_transform == 'T4':
                    if 'imagenet' in args.dataset:
                        from utils.augmix_im import augmix
                    else:
                        from utils.augmix import augmix
                    x_orig = [transforms.ToPILImage()(x_orig[i]) for i in range(len(x_orig))]
                    x_aug = [augmix(x_orig[i]) for i in range(len(x_orig))]
                    x_aug = torch.stack(x_aug, dim=0)
                    x_aug = normalize(args.dataset)(x_aug)
                    z_aug = ext(x_aug.cuda())

                cf = features_test[-len(z_aug):] - z_aug
                cfsets.append(cf)
                features_test2.append(z_aug)
            c = torch.mean(torch.stack(cfsets), dim=0)
            features_test2 = torch.mean(torch.stack(features_test2), dim=0)

            # Append and pop out c in queue
            if args.queue_size > args.batch_size:
                c_queue = queue_ext_c.get()
                queue_ext_c.update(c)
                feat2_queue = queue_ext_feat2.get()
                queue_ext_feat2.update(features_test2)
                if c_queue is not None:
                    c = torch.cat([c, c_queue.cuda()])
                    features_test2 = torch.cat([features_test2, feat2_queue.cuda()])
            # Obtain losses
            ## Information between z and c
            s_z, s_c = kernel_width(features_test.cpu().detach().numpy(), args.ks), kernel_width(c.cpu().detach().numpy(), args.ks)
            Izn = calculate_MI(features_test, c, s_z, s_c, args.alpha)
            classifier_loss += args.Izn_par*Izn

            if args.ent2:
                outputs_test2 = classifier(features_test2)
                softmax_out2 = nn.Softmax(dim=1)(outputs_test2)
                entropy_loss2 = torch.mean(Entropy(softmax_out2))
                classifier_loss += args.ent2_par*entropy_loss2
            if args.uncert:
                outputs_test2 = classifier(features_test2)
                softmax_out2 = nn.Softmax(dim=1)(outputs_test2)
                uncert = torch.mean(get_uncert(softmax_out2, softmax_out))
                uncert2 = torch.mean(get_uncert(softmax_out, softmax_out2))
                classifier_loss += args.uncert_par*(uncert+uncert2)
    
        classifier_loss.backward()
        optimizer.step()
        losses.update(classifier_loss.item(), labels.size(0))

    if args.dataset == 'visda':
        err_cls, _, _, err_each = test(teloader, net, args.dataset)
    else:
        err_cls = test(teloader, net, args.dataset)[0]
    all_err_cls.append(err_cls)

    toc = time.time()
    if args.dataset == 'visda':
        print(('Epoch %d/%d (%.0fs):' %(epoch, args.nepoch, toc-tic)).ljust(24) +
                        '%.2f\t\t' %(err_cls*100) +
                        '{loss.val:.4f}\t\t'.format(loss=losses) +
                        ' '.join(map(str, np.round(err_each.numpy()*100, 2))))
    else:
        print(('Epoch %d/%d (%.0fs):' %(epoch, args.nepoch, toc-tic)).ljust(24) +
                        '%.2f\t\t' %(err_cls*100) +
                        '{loss.val:.4f}'.format(loss=losses))

    # termination and save
    if epoch > args.stopepoch+1 and all_err_cls[-args.stopepoch] < min(all_err_cls[-args.stopepoch+1:]):
        print("Termination: {:.2f}".format(all_err_cls[-args.stopepoch]*100))
        # state = {'net': net.state_dict()}
        # save_file = os.path.join(args.outf, args.corruption + '_' +  args.method + '.pth')
        # torch.save(state, save_file)
        # print('Save model to', save_file)
        break

    # lr decay
    scheduler.step(err_cls)

# -------------------------------

prefix = os.path.join(args.outf, args.corruption + f'_{args.method}')
if args.tsne:
    feat_tar, label_tar, _ = visu_feat(ext, teloader, args.dataset, prefix+'_class.jpg')
    comp_feat(feat_src, label_src, feat_tar, label_tar, prefix+'_marginal.jpg')
    nuisance_tar, feat2_tar, nlabel_tar, _, _ = visu_nuisance(ext, teloader, args.dataset, os.path.join(args.outf, args.corruption + f'_tirnu1_{args.transform}.jpg'), args.transform)
    comp_feat(nuisance_tar, nlabel_tar, feat_tar, label_tar, os.path.join(args.outf, args.corruption + f'_nuisance_marginal1{args.transform}.jpg'))
    # tsne_all(feat_tar, label_src, feat2_tar, label_tar, nuisance_tar, nlabel_tar, os.path.join(args.outf, args.corruption + f'_feat2_marginal1{args.transform}.jpg'))
    # comp_feat(feat2_tar, nlabel_tar, feat_tar, label_tar, os.path.join(args.outf, args.corruption + f'_feat2_marginal1{args.transform}.jpg'))

# -------------------------------

# df = pd.DataFrame([all_err_cls]).T
# df.to_csv(prefix, index=False, float_format='%.4f', header=False)
