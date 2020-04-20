import argparse
import run

parser = argparse.ArgumentParser(description='NAS settings')

parser.add_argument('--dir', type=str, default='test', help='Experiment directory (default: test)')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default='datasets', help='path to datasets location (default: datasets)')

parser.add_argument('--training', type=int, default=1, help='1 means training supernet, 0 - evaluating nets')

parser.add_argument('--seed', type=int, default=1, metavar='seed', help='random seed (default: 1)')

parser.add_argument('--num_workers', type=int, default=8, metavar='num_workers', help='num of pytorch workers (default: 8)')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size (default: 256)')

parser.add_argument('--epochs', type=int, default=370, help='number of epochs to train (default: 370)')
parser.add_argument('--lr_init', type=float, default=0.1, help='initial learning rate for SGD (default: 0.1)')
parser.add_argument('--lr_start_cycle', type=int, default=250, help='epoch when cycling learning rate starts (default: 250)')
parser.add_argument('--cycle_period', type=int, default=30, help='period of cycling learning rate (default: 30)')

parser.add_argument('--first_net_id', type=int, default=0, required=False, help='first net architecture id to test one')
parser.add_argument('--last_net_id', type=int, default=5000000, required=False, help='last net architecture id to test one')

args = parser.parse_args()

print ('RUN PARAMS:')
for arg in vars(args):
    if args.training == 0:
        if arg not in ['epochs', 'lr_init', 'lr_start_cycle', 'cycle_period']:
           print (arg, getattr(args, arg))     
    else:
        if arg not in ['first_net_id', 'last_net_id']:
            print (arg, getattr(args, arg))

print('#'*50)

def main():
    run.run_evaluations(args)
    
if __name__ == '__main__':
    main()
