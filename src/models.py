import torch.nn as nn
import numpy as np
from cells import InvertedResidual, ConvBNReLU

class Identity(nn.Module):
    '''
    Identity cell
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class MultipleChoiceBlock(nn.Module):
    '''
    Block for supernet: includes all options for each cell
    '''

    def __init__(self, layers = [], in_channels = 32, n_choices = 2):
        super(MultipleChoiceBlock, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.n_choices = n_choices

    def forward(self, x, propagate):
        '''
        propagate is a binary matrix N_layers*N_cell_types
        1 in position [i,j] means selection cell j for position i
        '''

        output = 0
            
        for i in range(len(self.layers)):

            if propagate[i] == 0:
                continue
            output += self.layers[i](x)
            
        return output

def make_layers_supernet_full(first_conv_out_channels, propagate = [], put_downsampling = [], n_choices = 2):
    
    channels = first_conv_out_channels
    blocks = []
    for i in range(len(propagate)):
        
        if i in put_downsampling:
            blocks += [InvertedResidual(channels, channels*2, kernel_size=3, stride=2, expand_ratio=3, use_res_connect=0)]
            channels *= 2

        layers = [
            Identity(), #Identity
            InvertedResidual(channels, channels, kernel_size=3, stride=1, expand_ratio=3, use_res_connect=1),
            InvertedResidual(channels, channels, kernel_size=5, stride=1, expand_ratio=6, use_res_connect=1)
            ]

        blocks.append(MultipleChoiceBlock(layers = layers, in_channels = channels, n_choices = n_choices))


    return blocks, channels

class Supernet(nn.Module):
    '''
    Supernet model for training
    It has multiple choice blocks with all possible cells in all positions except downsampling ones
    '''

    def __init__(self, num_classes=10, batch_norm=True, propagate = [], put_downsampling = [4,8,12], first_conv_out_channels = 32, training = True, n_choices = 2):
        super(Supernet, self).__init__()

        #stem convolution
        self.in_conv = ConvBNReLU(3, first_conv_out_channels, kernel_size=3, stride=1)
        
        #main body (lsit of cells)
        blocks, out_channels = make_layers_supernet_full(first_conv_out_channels = first_conv_out_channels, propagate = propagate, put_downsampling = put_downsampling, n_choices = n_choices)
        self.blocks_list = nn.ModuleList(blocks)

        #simple classifier
        self.features_mixing = ConvBNReLU(out_channels, 1280, kernel_size=1, stride=1)
        self.out1 = nn.AdaptiveAvgPool2d((1, 1))
        self.out2 = nn.Linear(1280, num_classes)

        self.n_layers = len(propagate)
        self.n_choices = n_choices
        self.sample_random_paths = training #in training we propagate through one path, uniformly sampled
        self.propagate = propagate

    def forward(self, x):

        x = self.in_conv(x)
        
        mixed_blocks_cnt = 0
        for block in self.blocks_list:

            if isinstance(block, MultipleChoiceBlock):

                if self.sample_random_paths:
                    choice = np.random.randint(self.n_choices)
                    propagate = [0 for i in range(self.n_choices)]                
                    propagate[choice] = 1
                else:
                    propagate = self.propagate[mixed_blocks_cnt]

                x = block(x, propagate)
                
                mixed_blocks_cnt += 1

            else:

                x = block(x)

        x = self.features_mixing(x)
        x = self.out1(x)
        x = x.view(x.shape[0],-1)
        x = self.out2(x)        
        return x
################################################

def make_layers_supernet_single_path(first_conv_out_channels, propagate = [], put_downsampling = []):
    
    channels = first_conv_out_channels
    blocks = []

    for i in range(len(propagate)):
        
        if i in put_downsampling:
            blocks += [InvertedResidual(channels, channels*2, kernel_size=3, stride=2, expand_ratio=3, use_res_connect=0)]
            channels *= 2

        if propagate[i][0] == 1:
            blocks.append(Identity())
        elif propagate[i][1] == 1:
            blocks.append(InvertedResidual(channels, channels, kernel_size=3, stride=1, expand_ratio=3, use_res_connect=1))
        elif propagate[i][2] == 1:        
            blocks.append(InvertedResidual(channels, channels, kernel_size=5, stride=1, expand_ratio=6, use_res_connect=1))

    return blocks, channels

class SinglePathSupernet(nn.Module):
    def __init__(self, num_classes=10, batch_norm=True, propagate = [], put_downsampling = [4,8,12], first_conv_out_channels = 32):
        super(SinglePathSupernet, self).__init__()

        self.in_conv = ConvBNReLU(3, first_conv_out_channels, kernel_size=3, stride=1)
        
        blocks, out_channels = make_layers_supernet_single_path(first_conv_out_channels = first_conv_out_channels, propagate = propagate, put_downsampling = put_downsampling)
        self.blocks_list = nn.Sequential(*blocks)

        self.features_mixing = ConvBNReLU(out_channels, 1280, kernel_size=1, stride=1)

        self.out1 = nn.AdaptiveAvgPool2d((1, 1))
        self.out2 = nn.Linear(1280, num_classes)

        self.n_layers = len(propagate)
        self.propagate = propagate

    def forward(self, x):

        #stem convolution
        x = self.in_conv(x)        
        
        #main body (lsit of cells)
        x = self.blocks_list(x)

        #classifier
        x = self.features_mixing(x)
        x = self.out1(x)
        x = x.view(x.shape[0],-1)
        x = self.out2(x)        
        return x