import torch.nn as nn
import torch.nn.functional as F

from util import *

class cfr_net_nn(nn.Module):
    """
     Counter Factual Refression Neural Network similar to https://arxiv.org/abs/1606.03976
    """

    def __init__(self, configs):
        super(cfr_net_nn, self).__init__()

        self.D_in = configs['D_in']
        self.H_representation = configs['H_representation']
        self.H_regression = configs['H_regression']
        self.weight_init = configs['weight_init']
        self.n_in = configs['n_in']
        self.n_out = configs['n_out']
        self.p_lambda = configs['p_lambda']
        self.p_alpha = configs['p_alpha']

        ''' Initialize Exponential Linear Unit layer'''
        self.representation()
        self.regression()
        self.outcome()

        self.params_rep = []
        self.params_rep.extend(self.rep1.parameters())
        self.params_rep.extend(self.rep2.parameters())
        self.params_rep.extend(self.rep3.parameters())

        self.params_rest =  [p for p in self.parameters() if p not in set(self.params_rep)]

    def representation(self):
        self.rep1 = nn.Linear(self.D_in, self.H_representation)
        self.rep2 = nn.Linear(self.H_representation, self.H_representation)
        self.rep3 = nn.Linear(self.H_representation, self.H_representation)

    def regression(self):
        self.reg1_t = nn.Linear(self.H_representation, self.H_regression)
        self.reg2_t = nn.Linear(self.H_regression, self.H_regression)
        self.reg3_t = nn.Linear(self.H_regression, self.H_regression)
        self.reg1_c = nn.Linear(self.H_representation, self.H_regression)
        self.reg2_c = nn.Linear(self.H_regression, self.H_regression)
        self.reg3_c = nn.Linear(self.H_regression, self.H_regression)

    def outcome(self):
        self.linear_t = nn.Linear(self.H_regression, 1)
        self.linear_c = nn.Linear(self.H_regression, 1)


    def forward(self, data, counterfactual = False):
        """
        Implementation of forward pass of 3 fully-connected exponential-linear
        layers for the representation and 3 for the hypothesis (split in t=0 and t=1)

        Args:
            data: data of 1 experiment [T] of 672 patients [N] with 25 features [D]
            data['x']: input [N,D] of 1 experiment [T]
            data['t']: treatment (1) or control group (0) [N] of 1 experiment [T]
            data['y']: factual outcomes [N] of 1 experiment (evt. eruit halen)
            I: indexes of used data/patients [N] (train/validation)

        return:
            y_pred: output predictions of outcomes y
        """
        x = torch.Tensor(data['x'])    #when testing at [I,:,0]

        if counterfactual == True:
            y = torch.Tensor(data['ycf'])      #when testing [I,0]
            t = torch.Tensor(1 - data['t'])

        else:
            y = torch.Tensor(data['yf'])  # when testing [I,0]
            t = torch.Tensor(data['t'])  # when testing [I,0]

        '''Construct the 3 input/representation layers'''
        h = []                              #create empty list for 3 representation layers

        h.append(F.elu(self.rep1(x)))       #dim_x = 672, 25; dim_w = 25,200
        h.append(F.elu(self.rep2(h[0])))    #dim_h = 672, 200; dim_w = 200,200
        h.append(F.elu(self.rep3(h[1])))    #dim_h = 672,200; dim_w = 200,200

        '''Select output layer of representation network'''
        h_output_rep = h[len(h)-1]

        '''Normalize output layer'''
        h_output_rep = h_output_rep / safe_sqrt(torch.sum(torch.square(h_output_rep), 1, True))

        '''Compute the 3 output/regression layers (splitted in t=0 & t=1)
        1. Split output layers in t=0 and t=1'''
        it = torch.where(t > 0)[0]
        ic = torch.where(t < 1)[0]

        h_output_rep_t = h_output_rep[it, :]    # Gather slices of X that are treatment
        h_output_rep_c =  h_output_rep[ic, :]   # Gather slices of X that are control

        '''1.1. Compute output layers treatment (t=1) '''
        h_t = []

        h_t.append(F.elu(self.reg1_t(h_output_rep_t)))   #dim_h = 672,200; dim_w = 200,100
        h_t.append(F.elu(self.reg2_t(h_t[0])))
        h_t.append(F.elu(self.reg3_t(h_t[1])))

        h_output_t = h_t[-1]

        y_pred_t = self.linear_t(h_output_t)

        '''1.2. Compute output layers control (t=0) '''
        h_c = []

        h_c.append(F.elu(self.reg1_c(h_output_rep_c)))   #dim_h = 672,200; dim_w = 200,100
        h_c.append(F.elu(self.reg2_c(h_c[0])))
        h_c.append(F.elu(self.reg3_c(h_c[1])))

        h_output_c = h_c[-1]

        y_pred_c = self.linear_c(h_output_c)

        '''2. Combine outcome prediction y of treatment and control group again to one outcome prediction'''
        y_pred = torch.zeros_like(t)

        y_pred[it] = y_pred_t
        y_pred[ic] = y_pred_c

        return y_pred, h_output_rep

