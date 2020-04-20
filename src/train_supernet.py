import torch
import tabulate
import time
from torchsummary import summary

import utils
import models

def train_oneshot_model(args, data_loaders, n_cells, n_choices, put_downsampling = []):
    
    num_samples = utils.get_number_of_samples(args.dataset)

    device = 'cuda'
        
    utils.setup_torch(args.seed)

    print ('Initializing model...')

    #Create a supernet skeleton (include all cell types for each position)
    propagate_weights = [[1,1,1] for i in range(n_cells)]
    model_class = getattr(models, 'Supernet')
    
    #Create the supernet model  and its SWA ensemble version  
    model = model_class(num_classes=utils.get_number_of_classes(args.dataset), propagate=propagate_weights, training = True, 
        n_choices = n_choices, put_downsampling = put_downsampling).to(device)
    ensemble_model = model_class(num_classes=utils.get_number_of_classes(args.dataset), propagate=propagate_weights, training = True, 
        n_choices = n_choices, put_downsampling = put_downsampling).to(device)
    
    #These summaries are for verification purposes only
    #However, removing them will cause inconsistency in results since random generators are used inside them to propagate
    summary(model, (3, 32, 32), batch_size=args.batch_size, device='cuda')
    summary(ensemble_model, (3, 32, 32), batch_size=args.batch_size, device='cuda')

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=0.9,
        weight_decay=1e-4
    )

    start_epoch = 0

    columns = ['epoch time', 'overall training time', 'epoch', 'lr', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']

    lrs = []
    n_models = 0

    all_values = {}
    all_values['epoch'] = []
    all_values['lr'] = []

    all_values['tr_loss'] = []
    all_values['tr_acc'] = []
    
    all_values['val_loss'] = []
    all_values['val_acc'] = []   
    all_values['test_loss'] = []
    all_values['test_acc'] = []

    n_models = 0
    print ('Start training...')

    time_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()
        
        #lr = utils.get_cosine_annealing_lr(epoch, args.lr_init, args.epochs)
        lr = utils.get_cyclic_lr(epoch, lrs, args.lr_init, args.lr_start_cycle, args.cycle_period)
        utils.set_learning_rate(optimizer, lr)
        lrs.append(lr)
        
        train_res = utils.train_epoch(device, data_loaders['train'], model, criterion, optimizer, num_samples['train'])
        
        values = [epoch + 1, lr, train_res['loss'], train_res['accuracy']]
                   
        if (epoch+1) >= args.lr_start_cycle and (epoch + 1) % args.cycle_period == 0:

            all_values['epoch'].append(epoch+1)
            all_values['lr'].append(lr)
            
            all_values['tr_loss'].append(train_res['loss'])
            all_values['tr_acc'].append(train_res['accuracy'])

            val_res = utils.evaluate(device, data_loaders['val'], model, criterion, num_samples['val'])        
            test_res = utils.evaluate(device, data_loaders['test'], model, criterion, num_samples['test'])
            
            all_values['val_loss'].append(val_res['loss'])
            all_values['val_acc'].append(val_res['accuracy'])
            all_values['test_loss'].append(test_res['loss'])
            all_values['test_acc'].append(test_res['accuracy'])
            values += [val_res['loss'], val_res['accuracy'], test_res['loss'], test_res['accuracy']]
                        
            utils.moving_average_ensemble(ensemble_model, model, 1.0 / (n_models + 1))
            utils.bn_update(device, data_loaders['train'], ensemble_model)
            n_models += 1

            print (all_values)
            
        overall_training_time = time.time()-time_start
        values = [time.time()-time_ep, overall_training_time] + values
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        print(table)
    
    print('Training finished. Saving final nets...')
    utils.save_result(all_values, args.dir, 'model_supernet')           
    
    torch.save(model.state_dict(), args.dir + '/supernet.pth')
    torch.save(ensemble_model.state_dict(), args.dir + '/supernet_swa.pth')
