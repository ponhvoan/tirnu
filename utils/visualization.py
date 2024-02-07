import sys
sys.path.append("..")
import torch
import numpy as np
import torchvision.transforms.v2 as transforms
from utils.corruptions import corrupt
from utils.prepare_dataset import apply_lp_corruption, train_corruptions
import random 
from PIL import Image
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

########### Helper Functions ###########
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
#######----------------------------------------#######

def feat_tsne(feat, label, figname):
    tsne = TSNE(n_components=2).fit_transform(feat)
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)

    num_class = len(np.unique(label))
    if num_class == 10:
        cmap = 'tab10'
    elif num_class == 2:
        cmap = colors.ListedColormap(['red', 'blue'])
    elif num_class == 3:
        cmap = colors.ListedColormap(['blue', 'limegreen', 'red'])
    elif num_class == 12:
        cmap = 'Pastel1'
    else:
        raise NotImplementedError

    plt.scatter(tx, ty, c=label, s=3, cmap=cmap, alpha=0.2)
    plt.axis('square')
    plt.axis('off')

    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    plt.savefig(figname, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    print('Save tsne to {}'.format(figname))

    return tsne

def obtain_nuisance(ext, dataset, x_orig, features_test, no_iter=1, transform='T1'):
    cfsets = []
    feat2 = []
                
    for _ in range(no_iter):
        if transform == 'T1':
            x_aug = simclr_transforms(dataset)(x_orig)
            z_aug = ext(x_aug.cuda())
        elif transform == 'T2':
            # x_aug = apply_lp_corruption(x_orig, 8, combine_train_corruptions=True, train_corruptions=train_corruptions,
            #                         concurrent_combinations=1, max=False, noise='standard', epsilon=0)
            x_aug = apply_lp_corruption(x_orig, 8, combine_train_corruptions=False, train_corruptions=train_corruptions,
                                    concurrent_combinations=1, max=False, noise='uniform-l2', epsilon=0.25)
            x_aug = normalize(dataset)(x_aug)
            # x_aug = transforms.Normalize(mean=x_aug.mean(dim=(0,2,3)), std=x_aug.std(dim=(0,2,3)))(x_aug)
            z_aug = ext(x_aug.cuda())
        elif transform == 'T3':
            x_origT3 = torch.stack(transforms.PILToTensor()([transforms.ToPILImage()(x_orig[i]) for i in range(len(x_orig))]), dim=0)
            x_aug = []

            corruption = random.sample(['snow', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression'], k=1)[0]
            for i in range(len(x_origT3)):
                x_aug_i = transforms.ToTensor()(Image.fromarray(corrupt(x_origT3[i].permute(1,2,0).numpy(), corruption_name=corruption)).convert('RGB'))
                x_aug.append(x_aug_i)
            x_aug = torch.stack(x_aug, dim=0)
            x_aug = normalize(dataset)(x_aug)
            z_aug = ext(x_aug.cuda())
        elif transform == 'T4':
            if 'imagenet' in dataset:
                from utils.augmix_im import augmix
            else:
                from utils.augmix import augmix
            x_orig = [transforms.ToPILImage()(x_orig[i]) for i in range(len(x_orig))]
            x_aug = [augmix(x_orig[i]) for i in range(len(x_orig))]
            x_aug = torch.stack(x_aug, dim=0)
            x_aug = normalize(dataset)(x_aug)
            z_aug = ext(x_aug.cuda())

        cf = features_test - z_aug
        cfsets.append(cf)
        feat2.append(z_aug)
    nuisance = torch.mean(torch.stack(cfsets), dim=0)
    feat2 = torch.mean(torch.stack(feat2), dim=0)

    return nuisance, feat2

def visu_feat(encoder, dataloader, dataset, figname, num_sample=768):
    encoder.eval()
    if dataloader.batch_size >= num_sample:
        num_batch = 1
    else:
        num_batch, mod = divmod(num_sample, dataloader.batch_size)
        assert mod == 0, "Batch size error"
    stack_feat = list()
    stack_label = list()
    dl_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(num_batch):
            (inputs, _), labels = next(dl_iter)
            features = encoder(inputs.cuda())
            stack_feat.append(features.cpu().numpy())
            stack_label.append(labels.numpy())
    features_concat = np.concatenate(stack_feat)
    labels_concat = np.concatenate(stack_label)
    
    if not dataset=='cifar100':
        features_tsne_concat = feat_tsne(features_concat, labels_concat, figname)
        print('Save feature visualization to', figname)
        return features_concat, labels_concat, features_tsne_concat
    else:
        return features_concat, labels_concat

def visu_nuisance(encoder, dataloader, dataset, figname, transform='T1', num_sample=768):
    encoder.eval()
    if dataloader.batch_size >= num_sample:
        num_batch = 1
    else:
        num_batch, mod = divmod(num_sample, dataloader.batch_size)
        assert mod == 0, "Batch size error"
    stack_label = list()
    stack_c = list()
    stack_feat2 = list()
    dl_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(num_batch):
            (inputs, x_orig), labels = next(dl_iter)
            features = encoder(inputs.cuda())
            nuisance, feat2 = obtain_nuisance(encoder, dataset, x_orig, features, 1, transform)
            stack_label.append(labels.numpy())
            stack_c.append(nuisance.cpu().numpy())
            stack_feat2.append(feat2.cpu().numpy())
    labels_concat = np.concatenate(stack_label)
    nuisance_concat = np.concatenate(stack_c)
    feat2_concat = np.concatenate(stack_feat2)

    if not dataset=='cifar100':
        nuisance_tsne_concat = feat_tsne(nuisance_concat, labels_concat, figname[:-4] + '_nuisance' + figname[-4:])
        feat2_tsne_concat = feat_tsne(feat2_concat, labels_concat, figname[:-4] + '_feat2' + figname[-4:])
        print('Save nuisance visualization to', figname)
        return nuisance_concat, feat2_concat, labels_concat, nuisance_tsne_concat, feat2_tsne_concat
    else:
        return nuisance_concat, feat2_concat, labels_concat

def comp_feat(feat_src, label_src, feat_tar, label_tar, figname):
    features_concat = np.concatenate([feat_src, feat_tar])
    label_src[:] = 0
    label_tar[:] = 1
    labels_concat = np.stack([label_src, label_tar])
    feat_tsne(features_concat, labels_concat, figname)
    print('Save feature comparision to', figname)

def tsne_all(feat_tar, label_tar, feat2_tar, label_tar2, nuisance, label_nui, figname):
    features_concat = np.concatenate([feat_tar, feat2_tar, nuisance])
    label_tar[:] = 0
    label_tar2[:] = 1
    label_nui[:] = 2
    labels_concat = np.concatenate([label_tar, label_tar2, label_nui])
    feat_tsne(features_concat, labels_concat, figname)
    print('Save feature comparision to', figname)

#######----------------------------------------#######