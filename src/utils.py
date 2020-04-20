import numpy as np
import torch
import json
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import os
import shutil

def model_to_solution(model):
    solution = []
    for l in model:
        solution.append(str(l))

    return ''.join(solution)

def get_device(device_id):
    if device_id >= 0:
        device = torch.device('cuda:%d' % device_id)
    else:
        device = torch.device('cpu')
    return device

def save_result(result, dir, model_descirption):
    solution = model_to_solution(model_descirption)
    filename = '%s/evaluations/model_%s.json' % (dir, solution)
    json.dump(result, open(filename, 'w'))

def check_model_exist(dir, model_descirption):
    solution = model_to_solution(model_descirption)
    filename = '%s/evaluations/model_%s.json' % (dir, solution)
    return os.path.exists(filename)

def copy_solution(dir, model1_description, model2_description):
    solution1 = model_to_solution(model1_description)
    solution2 = model_to_solution(model2_description)
    
    filename1 = '%s/evaluations/model_%s.json' % (dir, solution1)
    filename2 = '%s/evaluations/model_%s.json' % (dir, solution2)
    
    shutil.copyfile(filename2, filename1)

def setup_torch(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_number_of_classes(dataset):
    if dataset == 'CIFAR10':
        return 10
    elif dataset == 'CIFAR100':
        return 100
    else:
        print('Dataset %s not implemented!' % dataset)
        exit(1)

def get_number_of_samples(dataset):
    if dataset == 'CIFAR10':
        return {'train':40000, 'val':10000, 'test':10000}
    if dataset == 'CIFAR100':
        return {'train':40000, 'val':10000, 'test':10000}
    else:
        print ('Dataset %s not implemented' % dataset)
        exit(1)

def get_number_of_trainable_params(model):
    total_params = sum([p.numel() for p in model.parameters()])
    return total_params

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_cyclic_lr(epoch, lrs, lr_init, lr_start_cycle, cycle_period):
    n_0 = lr_init
    l_0 = lr_start_cycle  + cycle_period

    if epoch < lr_start_cycle:         
        lr = 0.5 * n_0 * (1 + np.cos((np.pi * epoch) / l_0))
    else:
        lr = lrs[-cycle_period]

    return lr


#################Below this line the code is adopted from other project
#Copyright (c) 2018, Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
#All rights reserved.

def moving_average_ensemble(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def train_epoch(device, loader, model, criterion, optimizer, n_samples):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / n_samples,
        'accuracy': correct / n_samples * 100.0,
    }


def evaluate(device, loader, model, criterion, n_samples):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()
    
    model.train()

    return {
        'loss': loss_sum / n_samples,
        'accuracy': correct / n_samples * 100.0,
    }  

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(device, loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.to(device, non_blocking=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    del momenta