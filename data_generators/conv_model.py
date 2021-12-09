import torch
import torch.nn as nn

from collections import OrderedDict
import random
import time

from pathlib import Path

import predicting_performance.metrics as metrics

from predicting_performance.stats_collector import StatsCollector
from predicting_performance.trainer import Trainer

from automl.utils.utils_datasets import make_and_check_dir, timeSince, report_times

from predicting_performance.data_generators.debug_model_gen import save_model_info

from predicting_performance.data_loader_cifar import get_cifar10_for_data_point_mdl_gen
from predicting_performance.data_point_models.custom_layers import Flatten

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Gene_data(nn.Module):
    def __init__(self, min_conv_n = 1, max_conv_n = 5, min_fc_n = 1, max_fc_n = 5, para_min = 40000,min_filter = 26,min_fc = 32, max_filter = 32, max_fc = 256):
        super().__init__()
        self.kernel_choice = [1,3,5,7,9,11]
        self.min_conv_n = min_conv_n
        self.max_conv_n = max_conv_n
        self.min_fc_n = min_fc_n
        self.max_fc_n = max_fc_n
        self.in_channels = 3
        self.H = 32
        self.W = 32
        self.conv_n = []
        self.fc_list = []
        self.filter_list = []
        self.kernel_list = []
        self.do_bn=False
        self.para_min = para_min
        self.para_max = para_min * 50
        self.conv_n = 2
        self.min_filter = min_filter
        self.min_fc = min_fc
        self.max_filter = max_filter
        self.max_fc = max_fc

    def generate_random_inputs(self):
        '''
        to randomly generate input of
        create_model_datapoint(in_channels,conv_n,fc_list=[],filter_list = [],kernel_list = [],do_bn=False):
        '''
        self.conv_n = random.randint(self.min_conv_n, self.max_conv_n) # number of conv layer
        fc_listumber = random.randint(self.min_fc_n, self.max_fc_n) # number of fullyconnected layer
        self.fc_list=[] # list to take fullyconnected layer
        self.filter_list = []
        self.kernel_list = []
        #filter_list: list of filter number.
        i = self.conv_n
        while i >=  0:
            filter_to_append = random.randint(self.min_filter, self.max_filter)
            self.filter_list.append(filter_to_append)
            i =i - 1
            # min_c_filter = max(min_filter,filter_to_append - 64)
            # max_c_filter = min(max_filter, filter_to_append + 64)



        i = fc_listumber
        while i > 0:
            fc_to_append = random.randint(self.min_fc, self.max_fc)
            self.fc_list.append(fc_to_append)
            i =i - 1
            # min_c_fc = max(min_fc,fc_to_append - 64)
            # max_c_fc = min(max_fc, fc_to_append + 64)
        #do_bn: true or false
        do_bn_number = random.choice([0,1])
        if do_bn_number == 0:
            self.do_bn = False
        else:
            self.do_bn = True
        # in_channels = 3


        # kernel_list: list of kernal's value
        i = 30
        self.kernel_list.append(3)
        i = i - 1
        # print(i)
        while i > 0:
            self.kernel_list.append(random.choice(self.kernel_choice))
            i =i - 1
        # fc_list=[]
        # print(kernel_list)
        # return  conv_n, fc_list,filter_list, kernel_list, do_bn

    def create_model_datapoint(self):
        '''
        to create a code that takes create a sequential  looks like
        Sequential(
              (conv0): Conv2d(3, 2, kernel_size=(5, 5), stride=(1, 1))
              (selu1): SELU()
              (conv2): Conv2d(2, 20, kernel_size=(5, 5), stride=(1, 1))
              (tanh3): Tanh()
              (conv4): Conv2d(20, 17, kernel_size=(5, 5), stride=(1, 1))
              (selu5): SELU()
              (flatten6): Flatten()
              (linear7): Linear(in_features=17, out_features=24, bias=True)
              (tanh8): Tanh()
              (linear9): Linear(in_features=24, out_features=3, bias=True)
            )
        '''
        countfilter = 0
        countfullyconnect = 0
        countkernel = 0
        fc_list_size = len(self.fc_list)
        countFC_linear = 0
        init_algorithm_list = []
        init_hyerparam_list = []
        init_choice = ['default','xavier_uniform','xavier_normal','He_uniform','He_normal']


        #decide the structure
        architecture_order = []
        while self.conv_n > 0:
            architecture_order.append('conv')
            # if self.do_bn == True:
            #     architecture_order.append('batchnorm')
            # architecture_order.append(random.choice(['relu','selu','tanh','leaky_relu']))
            if self.do_bn == True:
                architecture_order.append('batchnorm')
            architecture_order.append(random.choice(['relu','selu','tanh','leaky_relu']))
            # if self.do_bn == True:
            #     architecture_order.append('batchnorm')
            self.conv_n = self.conv_n - 1
        architecture_order.append('Flatten')
        while fc_list_size > 1:
            architecture_order.append('linear')
            # if self.do_bn == True:
            #     architecture_order.append('batchnorm')
            # architecture_order.append(random.choice(['relu','selu','tanh','leaky_relu']))
            if self.do_bn == True:
                architecture_order.append('batchnorm1D')
            architecture_order.append(random.choice(['relu','selu','tanh','leaky_relu']))
            # if self.do_bn == True:
            #     architecture_order.append('batchnorm1D')
            fc_list_size = fc_list_size - 1
        if fc_list_size == 1:
            architecture_order.append('linear')
        print(architecture_order)

        if len(self.filter_list) == 0:
            out_channels = random.randint(1,10)
        else:
            out_channels = self.filter_list[countfilter]
            countfilter = countfilter + 1


        ##build the random layers
        #Each layer is being put into an ordered dictionary for the sequential model
        model_architecture = OrderedDict()
        i = 0
        for layer in architecture_order:
            if layer == 'conv':
                kernel_size = self.kernel_list[countkernel]
                countkernel = countkernel  + 1
                while kernel_size > self.in_channels:
                    kernel_size = self.kernel_list[countkernel]
                    countkernel = countkernel  + 1

                model_architecture[f'conv{i}'] = nn.Conv2d(self.in_channels,out_channels,kernel_size)
                conv2 = nn.Conv2d(self.in_channels,out_channels,kernel_size)
                init_use = random.choice(init_choice)
                init_algorithm_list.append(init_use)
                if init_use == 'xavier_uniform':
                     nn.init.xavier_uniform_(conv2.weight, gain=nn.init.calculate_gain('relu'))
                     init_hyerparam_list.append(str("gain=calculate_gain('relu')"))
                elif init_use =='xavier_normal':
                    nn.init.xavier_normal_(conv2.weight)
                    init_hyerparam_list.append(torch.__version__)
                elif init_use =='He_uniform':
                    nn.init.kaiming_uniform_(conv2.weight, mode='fan_in', nonlinearity='relu')
                    init_hyerparam_list.append(str("mode='fan_in', nonlinearity='relu'"))
                elif init_use =='He_normal':
                    nn.init.kaiming_normal_(conv2.weight, mode='fan_out', nonlinearity='relu')
                    init_hyerparam_list.append(str("mode='fan_out', nonlinearity='relu'"))
                else:
                    init_hyerparam_list.append(torch.__version__)


                #make the outputs of a layer the inputs of the next layer
                self.in_channels = out_channels
                #randomly choose the outputs of the next layer
                out_channels = self.filter_list[countfilter]
                countfilter = countfilter + 1
                self.H = self.H - kernel_size + 1

                #W = W - self.in_channels + 1
                # print('-------------')
                # print('H: %d',H)
            elif layer == 'linear':

                model_architecture[f'linear{i}'] = nn.Linear(self.in_channels,out_channels)
                lin = nn.Linear(self.in_channels,out_channels)
                init_use = random.choice(init_choice)
                init_algorithm_list.append(init_use)
                if init_use == 'xavier_uniform':
                     nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain('relu'))
                     init_hyerparam_list.append(str("gain=calculate_gain('relu')"))
                elif init_use =='xavier_normal':
                    nn.init.xavier_normal_(lin.weight)
                    init_hyerparam_list.append(torch.__version__)
                elif init_use =='He_uniform':
                    nn.init.kaiming_uniform_(lin.weight, mode='fan_in', nonlinearity='relu')
                    init_hyerparam_list.append(str("mode='fan_in', nonlinearity='relu'"))
                elif init_use =='He_normal':
                    nn.init.kaiming_normal_(lin.weight, mode='fan_out', nonlinearity='relu')
                    init_hyerparam_list.append(str("mode='fan_out', nonlinearity='relu'"))
                else:
                    init_hyerparam_list.append(torch.__version__)

                self.in_channels = out_channels
                out_channels = self.fc_list[countfullyconnect]
                countfullyconnect = countfullyconnect + 1

            elif layer == 'maxpool':
                if self.kernel_list == -1:
                    kernel_size = random.choice([1,3,5,7])
                else:
                    kernel_size = self.kernel_list
                model_architecture[f'maxpool{i}'] = nn.MaxPool2d(kernel_size,kernel_size)
            elif layer == 'batchnorm':
                model_architecture[f'batchnorm{i}'] = nn.BatchNorm2d(self.in_channels)
            elif layer == 'batchnorm1D':
                model_architecture[f'batchnorm1D{i}'] = nn.BatchNorm1d(self.in_channels)
            elif layer == 'relu':
                model_architecture[f'relu{i}'] = nn.ReLU()
            elif layer == 'selu':
                model_architecture[f'selu{i}'] = nn.SELU()
            elif layer == 'tanh':
                model_architecture[f'tanh{i}'] = nn.Tanh()
            elif layer == 'leaky_relu':
                negative_slope = random.random()
                model_architecture[f'leaky_relu{i}'] = nn.LeakyReLU(negative_slope)
            elif layer == 'softmax':
                model_architecture[f'softmax{i}'] = nn.Softmax()
            elif layer == 'Flatten':
                model_architecture[f'flatten{i}'] = Flatten()
                self.in_channels = self.H * self.H * self.in_channels

            else:
                p = random.random()
                model_architecture[f'dropout{i}'] = nn.Dropout2d(p)
            # print(in_channels, out_channels)
            i += 1
        mdl = nn.Sequential(model_architecture).to(device) ## Model!

        return mdl, init_algorithm_list,init_hyerparam_list

def count_nb_params(net):
    '''
    count the number of number_parameters
    '''
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    return count


def local_test_train(i, data_path):
    '''
    Generates a debugging dataset to cifar10
    This is for use in local CPU to test if the code is working or not.
    Try-catch structure can be applied.
    '''
    success_nb = 0
    while success_nb < 1:
        try:
            gene = Gene_data()
            number_parameters = 0
            while number_parameters < gene.para_min or number_parameters > gene.para_max:
                gene.generate_random_inputs()
                mdl, init_algorithm_list,init_hyerparam_list= gene.create_model_datapoint()
                number_parameters = count_nb_params(mdl)
                print(init_algorithm_list)
                print(init_hyerparam_list)
                print(number_parameters)

            trainloader, valloader, testloader = get_cifar10_for_data_point_mdl_gen()
            print(mdl)
            ##
            start = time.time()
            mdl = mdl.to(device)
            epochs = 1
            optimizer = torch.optim.Adam(mdl.parameters())
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1.0)
            criterion = nn.CrossEntropyLoss()
            error_criterion = metrics.error_criterion
            init_params = list(mdl.parameters())
            stats_collector = StatsCollector()
            iterations, train_iterations = 1, 1 # CAREFUL it might be returining 1, for real production we want it to be greater than 1
            trainer = Trainer(trainloader,valloader, testloader, optimizer, scheduler, criterion, error_criterion, stats_collector, device)
            train_loss, train_error, val_loss, val_error, test_loss, test_error = trainer.train_and_track_stats(mdl, epochs, iterations=iterations, train_iterations=train_iterations)
            final_params = list(mdl.parameters())
            ## save data point
            how_long, hours = timeSince(start)
            print(f'hours = {hours}')
            print(f'{how_long}')
            mdl_name = f'debug_{i}'
            other_data = trainer.stats_collector.get_stats_dict({'error_criterion':error_criterion.__name__})
            batch_size_train = trainloader.batch_size
            batch_size_test = testloader.batch_size
            batch_size_val = valloader.batch_size
            save_model_info(data_path, mdl, init_params, final_params,
                train_loss, train_error, val_loss, val_error, test_loss, test_error,
                optimizer, epochs, criterion, error_criterion,
                hours, mdl_name, init_algorithm_list, init_hyerparam_list,
                batch_size_train, batch_size_val, batch_size_test,
                number_parameters,
                scheduler, other_data)
            success_nb =success_nb + 1
            print('Success')
        except Exception as e:
            print('FAIL')
            print(e)

def main(i,gene,data_path,epochs,mdl_name):
    '''
    The main train fucntion to be used on GPU.
    Have a try-catch structure.
    i: each call will run i times and save i data points.
    gene: an object of Gene_data. Can change the default parameters in this fucntion. The default value are:
     (para_min = 40000, min_filter = 26, min_fc = 32, max_filter = 32, max_fc = 256,min_num_brick = 3, max_num_brick=6,max_para_times = 50, default_init_w_algor = True)
     data
    data_path: the root of where to save the results of training.
    epochs: number of epochs to train.
    mdl_name: change the mdl name.
    '''
    # get model type
    success_nb = 0
    while success_nb < 1:
        try:
            gene =Gene_data()
            number_parameters = 0
            while number_parameters < gene.para_min or number_parameters > gene.para_max:
                gene.generate_random_inputs()
                mdl, init_algorithm_list,init_hyerparam_list= gene.create_model_datapoint()
                number_parameters = count_nb_params(mdl)
                print(number_parameters)

            trainloader, valloader, testloader = get_cifar10_for_data_point_mdl_gen()
            ## create directory to save models
            make_and_check_dir(data_path)
            ## start creating models and its variations
            start = time.time()
            ## generate mdl data point
            mdl = mdl.to(device)
            optimizer = torch.optim.Adam(mdl.parameters())
            criterion = nn.CrossEntropyLoss()
            error_criterion = metrics.error_criterion
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1.0)
            stats_collector = StatsCollector()
            trainer = Trainer(trainloader,valloader, testloader, optimizer, scheduler, criterion, error_criterion, stats_collector, device)
            init_params = list(mdl.parameters())
            train_loss, train_error, val_loss, val_error, test_loss, test_error = trainer.train_and_track_stats(mdl, epochs)
            final_params = list(mdl.parameters())
            ## save data point
            how_long, seconds, minutes, hours = report_times(start)
            print(f'hours = {hours}')
            print(how_long)
            # mdl_name = f'tower_mdl_{i}'
            other_data = trainer.stats_collector.get_stats_dict({'error_criterion':error_criterion.__name__})
            batch_size_train = trainloader.batch_size
            batch_size_test = testloader.batch_size
            batch_size_val = valloader.batch_size
            save_model_info(data_path, mdl, init_params, final_params,
                train_loss, train_error, val_loss, val_error, test_loss, test_error,
                optimizer, epochs, criterion, error_criterion,
                hours, mdl_name, init_algorithm_list, init_hyerparam_list,
                batch_size_train, batch_size_val, batch_size_test,
                number_parameters,
                scheduler, other_data)
            success_nb =success_nb + 1
            print('Success')
        except Exception as e:
            print('FAIL')
            print(e)

if __name__ == '__main__':
    '''
    change those parameter before train:
    i: each call will run i times and save i data points.
    gene: an object of Gene_data. Can change the default parameters in this fucntion.The default value are:
     (para_min = 40000, min_filter = 26, min_fc = 32, max_filter = 32,
     max_fc = 256,min_num_brick = 3, max_num_brick=6,max_para_times = 50,
     default_init_w_algor = True,flag = False))
     data
    data_path: the root of where to save the results of training.
    epochs: number of epochs to train.
    mdl_name: change the mdl name.
    '''

    gene =Gene_data()
    data_path = '~/predicting_generalization/automl/data/Grp15_conv_fc'
    data_path = Path(data_path).expanduser()
    for i in range(1000):
        epochs = random.randint(300,550)
        mdl_name = f'conv_fc_{i}'
        main(i,gene,data_path,epochs,mdl_name)
    print('done \a')

    # #if want to test one local one:
    # data_path = '~/predicting_generalization/automl/data/xiao_test_test'
    # data_path = Path(data_path).expanduser()
    # for i in range(1):
    #     local_test_train(i,data_path)
    # print('done \a')
