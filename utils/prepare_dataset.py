import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np
import os
def prepare_transforms(dataset):

    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == 'visda':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dataset == 'imagenet-a' or dataset == 'imagenet-r':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError

    normalize = transforms.Normalize(mean=mean, std=std)

    if 'cifar' in dataset:
        size = 32
        te_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    elif dataset in ['visda', 'imagenet', 'imagenet-a', 'imagenet-r']:
        size = 224
        te_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(size),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])
    tr_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    simclr_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),     # TODO: modify the hard-coded size
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])

    return tr_transforms, te_transforms, simclr_transforms

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TestTransform:
    """Pass a normalized x and an unaugmented x"""
    def __init__(self, transform, crop=False):
        self.transform = transform
        self.crop = crop

    def __call__(self, x):
        if self.crop:
            crop_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
            return [self.transform(x), crop_transform(x)]
        else:
            return [self.transform(x), transforms.ToTensor()(x)]

# -------------------------

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_mix_corruption(args, num_mix, foldername):
    tesize = 10000
    if num_mix == 10:
        ## Contrast
        teset_c_raw = np.load(foldername + '/contrast.npy')
        teset_c_raw = teset_c_raw[(args.level-1)*tesize: args.level*tesize][:1000]

        ## De_focus
        teset_d_raw = np.load(foldername + '/defocus_blur.npy')
        teset_d_raw = teset_d_raw[(args.level-1)*tesize: args.level*tesize][1000:2000]

        ## Elastic
        teset_e_raw = np.load(foldername + '/elastic_transform.npy')
        teset_e_raw = teset_e_raw[(args.level-1)*tesize: args.level*tesize][2000:3000]

        ## Fog
        teset_f_raw = np.load(foldername + '/fog.npy')
        teset_f_raw = teset_f_raw[(args.level-1)*tesize: args.level*tesize][3000:4000]

        # ## Frost
        # teset_fr_raw = np.load(foldername + '/frost.npy')
        # teset_fr_raw = teset_fr_raw[(args.level-1)*tesize: args.level*tesize][4000:5000]

        ## Brightness
        teset_fr_raw = np.load(foldername + '/brightness.npy')
        teset_fr_raw = teset_fr_raw[(args.level-1)*tesize: args.level*tesize][4000:5000]

        ## Gaussian
        teset_g_raw = np.load(foldername + '/gaussian_noise.npy')
        teset_g_raw = teset_g_raw[(args.level-1)*tesize: args.level*tesize][5000:6000]

        ## Glass
        teset_gl_raw = np.load(foldername + '/glass_blur.npy')
        teset_gl_raw = teset_gl_raw[(args.level-1)*tesize: args.level*tesize][6000:7000]

        ## Impulse
        teset_i_raw = np.load(foldername + '/impulse_noise.npy')
        teset_i_raw = teset_i_raw[(args.level-1)*tesize: args.level*tesize][7000:8000]

        ## JPEG
        teset_j_raw = np.load(foldername + '/jpeg_compression.npy')
        teset_j_raw = teset_j_raw[(args.level-1)*tesize: args.level*tesize][8000:9000]

        ## Motion
        teset_m_raw = np.load(foldername + '/motion_blur.npy')
        teset_m_raw = teset_m_raw[(args.level-1)*tesize: args.level*tesize][9000:]

        teset_mix_raw = np.concatenate([teset_c_raw, teset_d_raw, teset_e_raw, teset_f_raw, teset_fr_raw, teset_g_raw, teset_gl_raw, teset_i_raw, teset_j_raw, teset_m_raw])

    elif num_mix == 5:
        # ## Frost
        # teset_fr_raw = np.load(foldername + '/frost.npy')
        # teset_fr_raw = teset_fr_raw[(args.level-1)*tesize: args.level*tesize][:2000]

        ## Snow
        teset_fr_raw = np.load(foldername + '/snow.npy')
        teset_fr_raw = teset_fr_raw[(args.level-1)*tesize: args.level*tesize][:2000]

        ## Gaussian
        teset_g_raw = np.load(foldername + '/gaussian_noise.npy')
        teset_g_raw = teset_g_raw[(args.level-1)*tesize: args.level*tesize][2000:4000]

        ## Glass
        teset_gl_raw = np.load(foldername + '/glass_blur.npy')
        teset_gl_raw = teset_gl_raw[(args.level-1)*tesize: args.level*tesize][4000:6000]

        ## Impulse
        teset_i_raw = np.load(foldername + '/impulse_noise.npy')
        teset_i_raw = teset_i_raw[(args.level-1)*tesize: args.level*tesize][6000:8000]

        ## JPEG
        teset_j_raw = np.load(foldername + '/jpeg_compression.npy')
        teset_j_raw = teset_j_raw[(args.level-1)*tesize: args.level*tesize][8000:]

        teset_mix_raw = np.concatenate([teset_fr_raw, teset_g_raw, teset_gl_raw, teset_i_raw, teset_j_raw])
    else:
        raise NotImplementedError

    return teset_mix_raw

# def create_symlinks_to_imagenet(imagenet_folder):
#     imagenet_val_location = "/home/voan/ttt-plus-plus/cifar/data/imagenet-r"
#     imagenet_r_wnids = {'n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677'}
#     if not os.path.exists(imagenet_folder):
#         os.makedirs(imagenet_folder)
#         folders_of_interest = imagenet_r_wnids  # os.listdir(folder_to_scan)
#         for folder in folders_of_interest:
#             os.symlink(imagenet_val_location + folder, imagenet_folder+folder, target_is_directory=True)
#     else:
#         print('Folder containing IID validation images already exists')

def prepare_test_data(args, ttt=False, num_sample=None):

    tr_transforms, te_transforms, simclr_transforms = prepare_transforms(args.dataset)

    if args.dataset == 'cifar10':

        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            print('Test on the original test set')
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                train=False, download=True, transform=TestTransform(te_transforms))
        elif args.corruption in common_corruptions:
            print('Test on %s level %d' %(args.corruption, args.level))
            teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
            teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                train=False, download=True, transform=TestTransform(te_transforms))
            teset.data = teset_raw

        elif args.corruption == 'cifar_new':
            from utils.cifar_new import CIFAR_New
            print('Test on CIFAR-10.1')
            teset = CIFAR_New(root=args.dataroot + '/CIFAR-10.1/datasets', transform=TestTransform(te_transforms))
            permute = False

        elif args.corruption == 'cifar_mix10':
            print('Test on mix on 10 noises on level %d' %(args.level))
            teset_mix_raw = prepare_mix_corruption(args, 10, args.dataroot + '/CIFAR-10-C')
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                train=False, download=True, transform=TestTransform(te_transforms))
            teset.data = teset_mix_raw

        elif args.corruption == 'cifar_mix5':
            print('Test on mix on 5 noises on level %d' %(args.level))
            teset_mix_raw = prepare_mix_corruption(args, 5, args.dataroot + '/CIFAR-10-C')
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                train=False, download=True, transform=TestTransform(te_transforms))
            teset.data = teset_mix_raw

        else:
            raise Exception('Corruption not found!')

    elif args.dataset == 'cifar100':
     
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            print('Test on the original test set')
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                                train=False, download=True, transform=TestTransform(te_transforms))
        elif args.corruption in common_corruptions:
            print('Test on %s level %d' %(args.corruption, args.level))
            teset_raw = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
            teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                                train=False, download=True, transform=TestTransform(te_transforms))
            teset.data = teset_raw

        elif args.corruption == 'cifar_mix5':
            print('Test on mix on 5 noises on level %d' %(args.level))
            teset_mix_raw = prepare_mix_corruption(args, 5, args.dataroot + '/CIFAR-100-C')
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                                train=False, download=True, transform=TestTransform(te_transforms))
            teset.data = teset_mix_raw

        elif args.corruption == 'cifar_mix10':
            print('Test on mix on 10 noises on level %d' %(args.level))
            teset_mix_raw = prepare_mix_corruption(args, 10, args.dataroot + '/CIFAR-100-C')
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                train=False, download=True, transform=TestTransform(te_transforms))
            teset.data = teset_mix_raw

        else:
            raise Exception('Corruption not found!')
    
    elif args.dataset == 'visda':
        from utils.visda import visda_dataset
        args.dataroot = 'data/visda'
        teset = visda_dataset(args.dataroot + '/validation', transform=TestTransform(te_transforms, crop=True))

    elif args.dataset == 'imagenet-r':
        args.dataroot = 'data/imagenet-r'
        teset = datasets.ImageFolder(root=args.dataroot, transform=TestTransform(te_transforms, crop=True))

    elif args.dataset == 'imagenet-a':
        args.dataroot = 'data/imagenet-a'
        teset = datasets.ImageFolder(root=args.dataroot, transform=TestTransform(te_transforms, crop=True))

    else:
        raise Exception('Dataset not found!')

    if not hasattr(args, 'workers') or args.workers < 2:
        pin_memory = False
    else:
        pin_memory = True

    if ttt:
        shuffle = True
        drop_last = True
    else:
        shuffle = True
        drop_last = False

    if args.dataset != 'visda' and 'imagenet' not in args.dataset:
        if num_sample and num_sample < teset.data.shape[0]:
            teset.data = teset.data[:num_sample]
            print("Truncate the test set to {:d} samples".format(num_sample))
    else:
        if num_sample and num_sample < len(teset):
            teset.imgs = teset.imgs[:num_sample]
            print("Truncate the test set to {:d} samples".format(num_sample))
        
    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                            shuffle=shuffle, num_workers=args.workers,
                                            worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=drop_last)
    return teset, teloader

def prepare_train_data(args, num_sample=None):
    print('Preparing data...')
    
    tr_transforms, te_transforms, simclr_transforms = prepare_transforms(args.dataset)

    if args.dataset == 'cifar10':

        if hasattr(args, 'ssl') and args.ssl == 'contrastive':
            trset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                         train=True, download=True,
                                         transform=TwoCropTransform(simclr_transforms))
            if hasattr(args, 'corruption') and args.corruption in common_corruptions:
                print('Contrastive on %s level %d' %(args.corruption, args.level))
                tesize = 10000
                trset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
                trset_raw = trset_raw[(args.level-1)*tesize: args.level*tesize]   
                trset.data = trset_raw
            elif hasattr(args, 'corruption') and args.corruption == 'cifar_new':
                from utils.cifar_new import CIFAR_New
                print('Contrastive on CIFAR-10.1')
                trset_raw = CIFAR_New(root=args.dataroot + '/CIFAR-10.1/datasets/', transform=te_transforms)
                trset.data = trset_raw.data
            elif  hasattr(args, 'corruption') and args.corruption == 'cifar_mix10':
                print('Test on mix on 10 noises on level %d' %(args.level))
                trset_mix_raw = prepare_mix_corruption(args, 10, args.dataroot + '/CIFAR-10-C')
                trset.data = trset_mix_raw
            elif  hasattr(args, 'corruption') and args.corruption == 'cifar_mix5':
                print('Test on mix on 5 noises on level %d' %(args.level))
                trset_mix_raw = prepare_mix_corruption(args, 5, args.dataroot + '/CIFAR-10-C')
                trset.data = trset_mix_raw
            else:
                print('Contrastive on ciar10 training set')
        else:
            trset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                        train=True, download=True, transform=tr_transforms)
            print('Cifar10 training set')

    elif args.dataset == 'cifar100':
        if hasattr(args, 'ssl') and args.ssl == 'contrastive':
            trset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                         train=True, download=True,
                                         transform=TwoCropTransform(simclr_transforms))            
            if hasattr(args, 'corruption') and args.corruption in common_corruptions:
                print('Contrastive on %s level %d' %(args.corruption, args.level))
                tesize = 10000
                trset_raw = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
                trset_raw = trset_raw[(args.level-1)*tesize: args.level*tesize]   
                trset.data = trset_raw
            else:
                print('Contrastive on ciar10 training set')
        else:
            trset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                            train=True, download=True, transform=tr_transforms)
            print('Cifar100 training set')
    else:
        raise Exception('Dataset not found!')

    if not hasattr(args, 'workers') or args.workers < 2:
        pin_memory = False
    else:
        pin_memory = True

    if num_sample and num_sample < trset.data.shape[0]:
        trset.data = trset.data[:num_sample]
        print("Truncate the training set to {:d} samples".format(num_sample))

    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers,
                                            worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=True)
    return trset, trloader

#################################################################################################################################################
############ Adapted from https://github.com/Georgsiedel/Lp-norm-corruption-robustness/blob/master/experiments/data_transforms.py ###############
#################################################################################################################################################
from skimage.util import random_noise
import torch.distributions as dist
import re 

train_corruptions = np.array([
        #['standard', 0.0, False],
        ['uniform-linf', 0.005, False],
        ['uniform-linf', 0.01, False],
        ['uniform-linf', 0.02, False],
        ['uniform-linf', 0.03, False],
        ['uniform-linf', 0.04, False],
        ['uniform-linf', 0.06, False],
        ['uniform-linf', 0.08, False],
        ['uniform-linf', 0.1, False],
        ['uniform-linf', 0.12, False],
        ['uniform-linf', 0.15, False],
        ['uniform-l2', 0.25, False],
        ['uniform-l2', 0.5, False],
        ['uniform-l2', 0.75, False],
        ['uniform-l2', 1.0, False],
        ['uniform-l2', 1.5, False],
        ['uniform-l2', 2.0, False],
        ['uniform-l2', 2.5, False],
        ['uniform-l2', 3.0, False],
        ['uniform-l2', 4.0, False],
        ['uniform-l2', 5.0, False],
        ['uniform-l0-impulse', 0.005, True],
        ['uniform-l0-impulse', 0.01, True],
        ['uniform-l0-impulse', 0.015, True],
        ['uniform-l0-impulse', 0.02, True],
        ['uniform-l0-impulse', 0.03, True],
        ['uniform-l0-impulse', 0.04, True],
        ['uniform-l0-impulse', 0.06, True],
        ['uniform-l0-impulse', 0.08, True],
        ['uniform-l0-impulse', 0.1, True],
        ['uniform-l0-impulse', 0.12, True]
        ])

def sample_lp_corr_batch(noise_type, epsilon, batch, density_distribution_max):
    img_corr = torch.zeros(batch[0].size(), dtype=torch.float32)

    if noise_type == 'uniform-linf':
        if density_distribution_max == True:  # sample on the hull of the norm ball
            rand = np.random.random(img_corr.shape)
            sign = np.where(rand < 0.5, -1, 1)
            img_corr = sign * epsilon
            img_corr = torch.from_numpy(img_corr)
        else: #sample uniformly inside the norm ball
            img_corr = dist.Uniform(img_corr - epsilon, img_corr + epsilon).sample()
    elif noise_type == 'uniform-linf-brightness': #only max-distribution, every pixel gets same manipulation
        img_corr = random.choice([-epsilon, epsilon])
    elif noise_type == 'gaussian': #note that this has no option for density_distribution=max
        var = epsilon * epsilon
        img_corr = torch.tensor(random_noise(img_corr, mode='gaussian', mean=0, var=var, clip=True))
    elif noise_type == 'uniform-l0-salt-pepper': #note that this has no option for density_distribution=max
        num_pixels = round(epsilon * torch.numel(img_corr[0]))
        pixels = random.sample(range(torch.numel(img_corr[0])), num_pixels)
        for pixel in pixels:
            max_pixel = random.choice([0, 1])
            img_corr[0].view(-1)[pixel] = max_pixel
            img_corr[1].view(-1)[pixel] = max_pixel
            img_corr[2].view(-1)[pixel] = max_pixel
    elif noise_type == 'uniform-l0-impulse':
        num_pixels = round(epsilon * torch.numel(img_corr))
        if density_distribution_max == True:
            pixels = random.sample(range(torch.numel(img_corr)), num_pixels)
            for pixel in pixels:
                img_corr.view(-1)[pixel] = random.choice([-1, 1])
        else:
            for id, img in enumerate(batch):
                pixels = []
                x = torch.numel(img_corr)
                pixels.append(random.sample(range(x*id, x*(id+1)), num_pixels))
            for pixel in pixels:
                batch.view(-1)[pixel] = random.randint(0, 255) / 255
    elif 'uniform-l' in noise_type:  #Calafiore1998: Uniform Sample Generation in lp Balls for Probabilistic Robustness Analysis
        d = len(img_corr.ravel())
        lp = [float(x) for x in re.findall(r'-?\d+\.?\d*', noise_type)]  # extract Lp-number from args.noise variable
        lp = lp[0]
        u = np.random.gamma(1 / lp, 1, size=(np.array(img_corr).shape))  # image-sized array of Laplace-distributed random variables (distribution beta factor equalling Lp-norm)
        u = u ** (1 / lp)
        rand = np.random.random(np.array(img_corr).shape)
        sign = np.where(rand < 0.5, -1, 1)
        norm = np.sum(abs(u) ** lp) ** (1 / lp) # scalar, norm samples to lp-norm-sphere
        if density_distribution_max == True:
            r = 1 # 1 to leave the sampled points on the hull of the norm ball, to sample uniformly within use this: np.random.random() ** (1.0 / d)
        else: #uniform density distribution
            r = np.random.random() ** (1.0 / d)
        img_corr = epsilon * r * u * sign / norm  #image-sized corruption, epsilon * random radius * random array / normed
        img_corr = torch.from_numpy(img_corr)
    elif noise_type == 'standard':
        pass
    else:
        print('Unknown type of noise')

    batch_corr = batch + img_corr.unsqueeze(0)
    batch_corr = np.clip(batch_corr, 0, 1)  # clip values below 0 and over 1
    return batch_corr

def apply_lp_corruption(batch, minibatchsize, combine_train_corruptions, train_corruptions, concurrent_combinations, max, noise, epsilon):

    minibatches = batch.view(-1, minibatchsize, batch.size()[1], batch.size()[2], batch.size()[3])
    epsilon = float(epsilon)
        #for id, img in enumerate(batch):
        #    corruptions_list = random.sample(list(train_corruptions), k=concurrent_combinations)
        #    for x, (noise_type, train_epsilon, max) in enumerate(corruptions_list):
        #        train_epsilon = float(train_epsilon)
        #        img = sample_lp_corr_img(noise_type, train_epsilon, img, max)
        #    batch[id] = img
    for id, minibatch in enumerate(minibatches):
        if combine_train_corruptions == True:
            corruptions_list = random.sample(list(train_corruptions), k=concurrent_combinations)
            for x, (noise_type, train_epsilon, max) in enumerate(corruptions_list):
                train_epsilon = float(train_epsilon)
                minibatch = sample_lp_corr_batch(noise_type, train_epsilon, minibatch, max)
            minibatches[id] = minibatch
        else:
            minibatch = sample_lp_corr_batch(noise, epsilon, minibatch, max)
            minibatches[id] = minibatch
    batch = minibatches.view(-1, batch.size()[1], batch.size()[2], batch.size()[3])

    return batch