# Adapted from https://github.com/archy666/MEIB
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

def kernel_width(x, ks=3):
    k_x = squareform(pdist(x, 'euclidean'))
    sigma = np.mean(np.mean(np.sort(k_x[:,:ks], 1)))
    return sigma

# Denote the feature of x by z and the nuisance factor by c
def pairwise_distances(z):
    bn = z.shape[0]
    z = z.view(bn, -1)
    instances_norm = torch.sum(z ** 2, -1).reshape((-1, 1))
    return -2*torch.mm(z, z.t())+instances_norm+instances_norm.t()

def calculate_gram_mat(z, sigma):
    dist = pairwise_distances(z)
    return torch.exp(-dist / sigma)

def reyi_entropy(z, sigma, alpha):
    k = calculate_gram_mat(z, sigma)
    k = k / torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy

def joint_entropy(z, c, s_z, s_c, alpha):
    z = calculate_gram_mat(z, s_z)
    c = calculate_gram_mat(c, s_c)
    k = torch.mul(z, c)
    k = k / torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eigv = torch.abs(torch.linalg.eigh(k)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def calculate_MI(z, c, s_z, s_c, alpha):
    Hz = reyi_entropy(z, sigma=s_z, alpha=alpha)
    Hc = reyi_entropy(c, sigma=s_c, alpha=alpha)
    Hzc = joint_entropy(z, c, s_z, s_c, alpha=alpha)
    Izn = Hz + Hc - Hzc
    return Izn