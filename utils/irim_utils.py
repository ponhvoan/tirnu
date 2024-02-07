"""
Modified from
https://github.com/DequanWang/tent/blob/master/tent.py
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim
from utils.meib import kernel_width, calculate_MI
import torchvision.transforms.v2 as transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

inv_normalize_cifar10 = transforms.Normalize(
    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    std=[1/0.2023, 1/0.1994, 1/0.2010]
)

normalize_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))])

simclr_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

def replace_batch_norm(model):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with another norm

    Args:
        model: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    try:
        model = model.ext
    except AttributeError:
        pass

    for name, module in model.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(model, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    # num_channels = sub_layer.num_featsures
                    # first level of current layer or model contains a batch norm --> replacing.
                    # layer._modules[name] = torch.nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels).cuda()
                    weight, bias = model._modules[name].weight, model._modules[name].bias # save BN affine parameters
                    model._modules[name] = torch.nn.InstanceNorm2d(sub_layer.num_features, affine=True, track_running_stats=True, device='cuda')
                    model._modules[name].weight, model._modules[name].bias = weight, bias
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split('.')[0]
                sub_layer = getattr(model, name)
                sub_layer = replace_batch_norm(sub_layer)
                model.__setattr__(name=name, value=sub_layer)
    return model

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    #Forward and adapt model on batch of data.
    
    # Get outputs (logits) and extract features z
    outputs = model(x)
    # train_nodes, eval_nodes = get_graph_node_names(model)
    # print(train_nodes, eval_nodes)
    return_nodes = {"ext.flatten": "ext.flatten"}
    feat_extract = create_feature_extractor(model, return_nodes=return_nodes)
    z = feat_extract(x)['ext.flatten']
    x_orig = inv_normalize_cifar10(x)

    cfsets = []
    for _ in range(10):
        x_aug = simclr_transforms(x_orig)
        z_aug = feat_extract(x_aug.cuda())['ext.flatten']
        cf = z - z_aug
        cfsets.append(cf)
    c = torch.mean(torch.stack(cfsets), dim=0)
    # Obtain losses
    ## Information between z and c
    s_z, s_c = kernel_width(z.cpu().detach().numpy()), kernel_width(c.cpu().detach().numpy())
    Izc = calculate_MI(z, c, s_z, s_c)
    ## H(y) prediction entropy
    pred_entr = softmax_entropy(outputs).mean(0) 
    ## Overall objective
    loss = pred_entr + 0.1*Izc 
    
    # Update gradient
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of instance stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

        # for np, n in m.named_parameters():
        #     if 'ext.flatten' in np:
        #         break
        # else:
        #     continue
        # break
    return model

def freeze_head(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    model.requires_grad_(True)
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = False
    return model

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"

def setup_tent(model, args):
    """Set up tent adaptation.
    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    # replace_batch_norm(model)
    model = configure_model(model)
    # model = freeze_head(model)
    params, param_names = collect_params(model)
    optimizer = setup_optimizer(params, args)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
    tent_model = Tent(model, optimizer,
                      steps=1,
                      episodic=False)
    # print(f"model for adaptation: %s", model)
    # print(f"params for adaptation: %s", param_names)
    # print(f"optimizer for adaptation: %s", optimizer
    return tent_model

def setup_optimizer(params, args):
    """Set up optimizer for tent adaptation.
    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.
    For best results, try tuning the learning rate and batch size.
    """
    optimizer = optim.Adam(params,
                            lr=args.lr,
                            betas=(0.9, 0.999),
                            weight_decay=0.)
    return optimizer