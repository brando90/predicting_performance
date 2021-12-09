import torch
import torch.nn as nn

#datasets = ['cifar10', 'cifar100', 'MNIST']

class AutoMLVocab:

    def __init__(self):
        ## set architecture vocab data structures
        self.architecture_vocab, self.architecture2idx = self._init_architecture_vocab()
        ## set the data set vocab
        self.data_sets_vocab, self.data_sets2idx = self._init_data_sets()
        ##
        self.hparms_vocab, self.hptype2idx = self._init_hp_params()

    def _init_architecture_vocab(self):
        '''
        Initializes the architecture vocabulary
        '''
        #vocab = ['SOS','EOS', nn.Conv2d, nn.Linear, nn.ReLU, nn.SELU]
        #vocab = ['EOS', nn.Conv2d, nn.Linear, nn.ReLU, nn.SELU]
        vocab = ['EOS', nn.Conv2d, nn.Linear, nn.ReLU, nn.SELU, torch.nn.ELU]
        vocab2idx = { vocab[i]:i for i in range(len(vocab)) } # faster than using python's list.index(element)
        return vocab, vocab2idx

    def _init_data_sets(self):
        '''
        Initializes the data sets vocabulary
        '''
        #vocab = ['EOS', 'cifar10', 'cifar100', 'MNIST']
        vocab = ['cifar10']
        vocab2idx = { vocab[i]:i for i in range(len(vocab)) } # faster than using python's list.index(element)
        return vocab, vocab2idx

    def _init_hp_params(self):
        '''
        Initializes the hyperparameter architecture vocabulary
        '''
        vocab = [64,128,256, # Linear HPs = (out_features)
                16,32,64, (1,1),(3,3),(5,5) # Conv2d HPs = (out_channels,kernel_size)
                ]
        vocab2idx = { vocab[i]:i for i in range(len(vocab)) } # faster than using python's list.index(element)
        ## where the interval of indices where hps start (start_inclusive,end_exclusive)
        self.hp_start_indices = {
                nn.Linear:[(0,3)], # Linear HPs = (out_features)
                nn.Conv2d:[(3,6),(6,9)] # Conv2d HPs = (out_channels,kernel_size)
        }
        ##
        return vocab, vocab2idx
