import time
import torch
import numpy as np
from thop.profile import profile

import utils
import models

def run_evaluation(args, model, data_loaders, model_description, n_choices, layers_types, downsample_layers):
    
    start = time.time()
    
    num_samples = utils.get_number_of_samples(args.dataset)

    all_values = {}
    
    device = 'cuda'
    
    #setting up random seeds
    utils.setup_torch(args.seed)

    #creating model skeleton based on description
    propagate_weights = []
    for layer in model_description:
        cur_weights = [0 for i in range(n_choices)]
        cur_weights[layers_types.index(layer)] = 1
        propagate_weights.append(cur_weights)

    model.propagate = propagate_weights   

    #Create the computationally identical model but without multiple choice blocks (just a single path net)
    #This is needed to correctly measure MACs
    pruned_model = models.SinglePathSupernet(num_classes=utils.get_number_of_classes(args.dataset), propagate=propagate_weights, put_downsampling = downsample_layers)#.to(device)
    pruned_model.propagate = propagate_weights
    inputs = torch.randn((1,3,32,32))
    total_ops, total_params = profile(pruned_model, (inputs,), verbose=True)
    all_values['MMACs'] = np.round(total_ops / (1000.0**2), 2)
    all_values['Params'] = int(total_params)
    
    del pruned_model
    del inputs

    ################################################
    criterion = torch.nn.CrossEntropyLoss()    
    
    #Initialize batch normalization parameters
    utils.bn_update(device, data_loaders['train_for_bn_recalc'], model)

    val_res = utils.evaluate(device, data_loaders['val'], model, criterion, num_samples['val'])        
    test_res = utils.evaluate(device, data_loaders['test'], model, criterion, num_samples['test'])
           
    all_values['val_loss'] = np.round(val_res['loss'], 3)
    all_values['val_acc'] = np.round(val_res['accuracy'], 3)
    all_values['test_loss'] = np.round(test_res['loss'], 3)
    all_values['test_acc'] = np.round(test_res['accuracy'], 3)

    print (all_values, 'time taken: %.2f sec.' % (time.time()-start))

    utils.save_result(all_values, args.dir, model_description)