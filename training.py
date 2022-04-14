import torch.optim as optim
from tqdm import tqdm

import cfr_net_nn as CFR
from loss_function import *
from util import *
import pickle


def train(network, optimizer, D_exp, I_train, I_valid, D_exp_test, configs):
    """
    Training of the data set per experiment/repetition in batches

    inputs:
        - network: cfr_network used to compute layers, already initialized; cfr_net = CFR(configs)
        - optimizer: ADAM optimizer with weight decay for parameters of regression network
        - D_exp = training data of 1 experiment (i_exp)
        - I_train = indices of training set of training data (computed through validation_split)
        - I_valid = indices of test set of training data (computed through validation_split)
        - D_exp_test = test data set of 1 experiment (i_exp)

    return:
        - losses
    """

    batch_size = configs['batch_size']
    obj_loss = []

    ''' Compute data set dictionaries for training'''
    D_train_f, D_validation_f, D_train_cf, D_test_f, D_test_cf = create_data_dictionaries(D_exp, I_train, I_valid, D_exp_test)

    ''' Compute first predictions without optimization '''
    pred_train_f, h_output_rep_train_f = network.forward(D_train_f)

    ''' Compute first losses without optimization'''
    objective_loss_train, pred_train_error = \
        total_loss(network.p_alpha, D_train_f['t'], D_train_f['yf'], pred_train_f, h_output_rep_train_f, configs)
    obj_loss.append(objective_loss_train)

    ''' Iterate over batches '''
    for batch in batches(I_train, batch_size):
        ''' Create the data dictionary with size batch_size of the full data train set'''
        D_train_f_batch, _, _, _, _ = create_data_dictionaries(D_exp, batch, None, None)

        ''' Zero the parameter gradients'''
        optimizer.zero_grad()

        ''' Compute outputs of the batch using forward '''
        pred_train_batch, h_output_batch = network.forward(D_train_f_batch)

        ''' Compute losses of the batch'''
        objective_loss_batch, pred_batch_error = \
            total_loss(network.p_alpha, D_train_f_batch['t'], D_train_f_batch['yf'], pred_train_batch, h_output_batch, configs)

        obj_loss.append(objective_loss_batch)
        ''' Backward propagation and optimization of the parameters'''
        objective_loss_batch.backward()
        optimizer.step()

    return obj_loss

def run(data_train, data_test, configs):
    '''Create instance of CFR Network'''
    cfr_net = CFR.cfr_net_nn(configs)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device used: ', DEVICE)
    cfr_net = cfr_net.to(DEVICE)

    '''
    Create ADAM optimizer with: lr = 0.05 & decay_rate = 0.97
    Apply weight decay to the regression layers'''
    lr = 0.05
    decay_rate = 0.97
    opt = optim.Adam([
        {'params': cfr_net.params_rep},
        {'params': cfr_net.params_rest, 'weight_decay':cfr_net.p_lambda}
        ], lr=lr)

    '''Set up exponential decay scheduler'''
    exp_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=decay_rate)

    ''' Set up losses and predictions for storing'''
    all_preds_train = []
    all_preds_test = []
    all_obj_losses = []
    all_I_valid = []

    for i_exp in range(1, configs['n_experiments']+1):
        data_exp, data_test_exp = data_per_experiment(data_train, data_test, i_exp)
        I_train, I_valid = validation_split(data_exp)

        '''Store indices of validation set of experiment i_exp'''
        all_I_valid.append(I_valid)

        '''Set the number of epochs for training, total 3000 iterations per experiment
        (5 batches and 600 times looping over the dataset)'''
        epochs = configs['epochs']

        '''Set up for storing objective losses per batch and predictions per 200 iterations (40 epochs * 5 batches)'''
        obj_losses = []
        preds_train = []
        preds_test = []

        '''Loop over the dataset multiple times'''
        for epoch in tqdm(range(epochs)):

            obj_loss = train(cfr_net, opt, data_exp, I_train, I_valid, data_test_exp, configs)

            '''Append objective loss of the training batches per epoch'''
            obj_losses.append(obj_loss)

            '''Determine predictions for every 40 iterations (5 batches * 40 epochs = 200 = output delay of example)'''
            if epoch % 20 == 0:
                '''Exponential weight decay scheduler step every 20 epochs'''
                exp_scheduler.step()

                with torch.no_grad():
                    pred_train_f, _ = cfr_net.forward(data_exp)
                    pred_train_cf, _ = cfr_net.forward(data_exp, counterfactual=True)
                    preds_train.append(torch.cat((pred_train_f, pred_train_cf), 1))

                    pred_test_f, _ = cfr_net.forward(data_test_exp)
                    pred_test_cf, _ = cfr_net.forward(data_test_exp, counterfactual=True)
                    preds_test.append(torch.cat((pred_test_f, pred_test_cf), 1))

        '''Store the objective loss of training set batches and the train and test predictions per experiment'''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_obj_losses.append(obj_losses)

        intermediate_result = {'train': all_preds_train, "test": all_preds_test, 'losses': all_obj_losses,
                               "I_valid": all_I_valid}

        '''Save intermediate result'''
        file = open("Results/Final_result/intermediate.pkl", "wb")
        pickle.dump(intermediate_result, file)
        file.close()

    print('\nFinished.')

    return all_preds_train, all_preds_test, all_obj_losses, all_I_valid
