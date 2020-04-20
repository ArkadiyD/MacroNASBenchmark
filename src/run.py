import argparse
import os
import pickle
import numpy as np
import torchvision
import torch
import itertools
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

import cifar
import train_supernet
import models
import utils
import evaluate_net

#############################
#Models Configuration    

#'I' is identity cell, '1' is MBConv 3,3x3, '2' is MBConv 6,5x5
CELL_TYPES = ['I','1','2']
N_CELLS = 14 #number of searchable cells
N_CHOICES = len(CELL_TYPES) #3 choices for each cell
DOWNSAMPLE_LAYERS = [4,8,12] #Downsampling cells (fixed) are placed after searchable cells with numbers 4,8,12 (enumerating from 0)
#############################

def run_evaluations(args):
    
    print('Making directory %s' % args.dir)
    os.makedirs(args.dir, exist_ok=True)
    os.makedirs(args.dir+'/evaluations', exist_ok=True)

    #We'll be loading datasets from torchvision    
    print('Loading dataset %s from %s' % (args.dataset, args.data_path))
    ds = getattr(torchvision.datasets, args.dataset)
    path = os.path.join(args.data_path, args.dataset.lower())

    if args.dataset == 'CIFAR10':
        transforms_dict = cifar.get_cifar10_transforms()
    elif args.dataset == 'CIFAR100':
        transforms_dict = cifar.get_cifar100_transforms()    
    else:
        print ('Dataset %s not implemented' % args.dataset)
        exit(1)

    
    #Download and plug in necessary transforms
    train_set = ds(path, train=True, download=True, transform=transforms_dict['train'])
    val_set = ds(path, train=True, download=True, transform=transforms_dict['test'])
    test_set = ds(path, train=False, download=True, transform=transforms_dict['test'])

    np.random.seed(42)

    #If using for the first time
    if not os.path.exists(os.path.join(args.dir, 'data_split.pkl')):

        #train/val stratified split, val size is 10000
        stratified_targets_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_indices, val_indices in stratified_targets_split.split(train_set.targets, train_set.targets):
            print("TRAIN:", len(train_indices))
            print("VAL:", len(val_indices))
        
        #10000 from train are selected for recalculating batch normalization coefficients during nets evaluations
        stratified_targets_split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)    
        train_targets = np.array(train_set.targets)[np.array(train_indices)]
        for _, train_for_bn_indices in stratified_targets_split.split(train_targets, train_targets):
            train_for_bn_indices = train_indices[train_for_bn_indices]
            print("TRAIN FOR BN:", len(train_for_bn_indices))
        
        indices = {'train':train_indices, 'val':val_indices, 'train_for_bn':train_for_bn_indices}
    
        pickle.dump(indices, open(os.path.join(args.dir, 'data_split.pkl'), 'wb'))

    else:
        indices = pickle.load(open(os.path.join(args.dir, 'data_split.pkl'), 'rb'))
        train_indices, val_indices, train_for_bn_indices = indices['train'], indices['val'], indices['train_for_bn']

    train_targets = np.array(train_set.targets)[train_indices]
    val_targets = np.array(train_set.targets)[val_indices]
    train_for_bn_targets = np.array(train_set.targets)[train_for_bn_indices]
    
    #Making sure that class proportions in all subsets are equal
    print ('Number of samples per class in train:',Counter(train_targets))
    print ('Number of samples per class in val:',Counter(val_targets))
    print ('Number of samples per class in samples for batch norm recalculation:',Counter(train_for_bn_targets))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    train_to_bn_sampler = torch.utils.data.SubsetRandomSampler(train_for_bn_indices)

    data_loaders = {
        'val': torch.utils.data.DataLoader(
            val_set, #without augmentations
            batch_size=args.batch_size, 
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        ),

        'test': torch.utils.data.DataLoader(
            test_set,  #without augmentations
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True),

        'train_for_bn_recalc': torch.utils.data.DataLoader(
            val_set, #without augmentations
            batch_size=args.batch_size,
            sampler=train_to_bn_sampler,
            num_workers=args.num_workers,
            pin_memory=True)
    }
    
    #Train data is needed only for supernet training
    if args.training:
        data_loaders['train']= torch.utils.data.DataLoader(
            train_set, #with_augmentations
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

    
    if args.training: #supernet creation and training
        train_supernet.train_oneshot_model(args, data_loaders, N_CELLS, N_CHOICES, put_downsampling = DOWNSAMPLE_LAYERS)

    else:
        print ('Initializing models...')
        all_possible_nets = []
        for length in range(N_CELLS, N_CELLS + 1):
            all_possible_nets += list(itertools.product(CELL_TYPES, repeat = length))
        all_possible_nets = sorted(all_possible_nets)

        np.random.seed(42) #this seed is used for ordering all nets only
        all_possible_nets = np.random.permutation(all_possible_nets)
        all_possible_nets = all_possible_nets[args.first_net_id : args.last_net_id]
        
        #model initialization
        model_class = models.Supernet
        model = model_class(num_classes=utils.get_number_of_classes(args.dataset), propagate=[[1 for i in range(N_CHOICES)] for j in range(N_CELLS)], 
           training = False, n_choices = N_CHOICES, put_downsampling = DOWNSAMPLE_LAYERS)
        
        #uploading weights from trained supernet (ensemble is for weights obtained with SWA)
        if not os.path.exists(args.dir + '/supernet_swa.pth'):
            print ('You need to train the supernet first!')
            exit(1)

        model.load_state_dict(torch.load(args.dir + '/supernet_swa.pth'))
        model.cuda()

        for i, model_description in enumerate(all_possible_nets):

            if i % 10000 == 0 and i > 0:
                torch.cuda.empty_cache()

            print ('evaluating genotype #%d:' % i, ''.join(model_description))
            
            #Check if the same model already evaluated
            if utils.check_model_exist(args.dir, model_description):
                print('model already evaluated')
                continue

            #Translate the genotype to minimal computationally equivalent genotype (omitting identity cells)
            real_model_description = []
            for l in range(N_CELLS):
                if l in DOWNSAMPLE_LAYERS:
                    real_model_description.append('D')
                if model_description[l] != 'I': #not indentity
                    real_model_description.append(model_description[l])
            
            print('real model to be computed:', ''.join(real_model_description))
            
            #Check if the identical (computationally) model already evaluated
            if utils.check_model_exist(args.dir, real_model_description):
                print ('identical model already evaluated')
                utils.copy_solution(args.dir, model_description, real_model_description)
                continue

            try:
                evaluate_net.run_evaluation(args, model, data_loaders, model_description, N_CHOICES, CELL_TYPES, DOWNSAMPLE_LAYERS)
            except RuntimeError as e:
                print (e)
                continue
                
            try:
                #copy from current genotype to real (after removing identities) genotype to re-use it
                utils.copy_solution(args.dir, real_model_description, model_description) 
            except Exception as e:
                print(e)

