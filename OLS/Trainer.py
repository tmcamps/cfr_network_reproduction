from Util_OLS import *
from OLS_nn_l1 import *
from OLS_nn_l2 import *
from OLS_evaluation import evaluate_result

from datetime import datetime
import torch.optim as optim
from tqdm import tqdm

def run(model, n_experiments, epochs, b_size, data_train, data_test):
    '''Setup l1 or l2 model'''
    if model == 'l1':
        OLSnet = OLSnet_l1()
    elif model == 'l2':
        OLSnet = OLSnet_l2()
    else:
        print('ERROR: Invalid model')

    # Create loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(OLSnet.parameters(), lr=5e-4)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device used: ', DEVICE)
    OLSnet.to(DEVICE)

    '''Create tensors from data'''
    treatment_train = torch.Tensor(data_train['t']).to(DEVICE)
    treatment_test = torch.Tensor(data_test['t']).to(DEVICE)

    tensors_train_f = torch.Tensor(data_train['x']).to(DEVICE), torch.Tensor(data_train['yf']).long().to(DEVICE)
    tensors_test_f = torch.Tensor(data_test['x']).to(DEVICE), torch.Tensor(data_test['yf']).long().to(DEVICE)

    tensors_train_cf = torch.Tensor(data_train['x']).to(DEVICE), torch.Tensor(data_train['ycf']).long().to(DEVICE)
    tensors_test_cf = torch.Tensor(data_test['x']).to(DEVICE), torch.Tensor(data_test['ycf']).long().to(DEVICE)

    '''Training loop'''
    all_pred_train = []
    all_pred_test = []

    if model == 'l1':
        print('Training l1 ...')
        for i in range(n_experiments):
            y_pred_train = []
            y_pred_test = []

            for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
                """Factual"""
                yf_preds_train = train_OLS_l1(tensors_train_f, OLSnet, optimizer, criterion, i, b_size, treatment_train[:,i])
                yf_preds_test = test_OLS_l1(tensors_test_f, OLSnet, criterion, i, b_size, treatment_test[:,i])

                """Counterfactual"""
                ycf_preds_train = train_OLS_l1(tensors_train_cf, OLSnet, optimizer, criterion, i, b_size, treatment_train[:,i])
                ycf_preds_test = test_OLS_l1(tensors_test_cf, OLSnet, criterion, i, b_size, treatment_test[:,i])

                y_cat_train = torch.cat((yf_preds_train[:, None], ycf_preds_train[:, None]), 1)
                y_cat_test = torch.cat((yf_preds_test[:, None], ycf_preds_test[:, None]), 1)

                y_pred_train.append(y_cat_train)
                y_pred_test.append(y_cat_test)

            all_pred_train.append(y_pred_train)
            all_pred_test.append(y_pred_test)

        print('Finished Training')
    
    elif model == 'l2':
        print('Training l2 ...')
        for i in range(n_experiments):
            y_pred_train = []
            y_pred_test = []

            for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
                """Factual"""
                yf_preds_train = train_OLS_l2(tensors_train_f, OLSnet, optimizer, criterion, i, b_size, treatment_train[:,i])
                yf_preds_test = test_OLS_l2(tensors_test_f, OLSnet, criterion, i, b_size, treatment_test[:,i])

                """Counterfactual"""
                ycf_preds_train = train_OLS_l2(tensors_train_cf, OLSnet, optimizer, criterion, i, b_size, treatment_train[:,i])
                ycf_preds_test = test_OLS_l2(tensors_test_cf, OLSnet, criterion, i, b_size, treatment_test[:,i])

                y_cat_train = torch.cat((yf_preds_train[:, None], ycf_preds_train[:, None]), 1)
                y_cat_test = torch.cat((yf_preds_test[:, None], ycf_preds_test[:, None]), 1)

                y_pred_train.append(y_cat_train)
                y_pred_test.append(y_cat_test)

            all_pred_train.append(y_pred_train)
            all_pred_test.append(y_pred_test)

        print('Finished Training')
    
    print('Evaluating results ...')

    '''Evaluation'''
    fake_val_id = np.arange(0,100)
    eval_train = evaluate_result(all_pred_train, data_train, fake_val_id, n_experiments)
    eval_test = evaluate_result(all_pred_test, data_test, fake_val_id, n_experiments)

    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    np.save('%s_%s.npy' % ('OLS/Results_OLS/OLS_'+model+'/IHDP.eval_train', time_now), eval_train)
    np.save('%s_%s.npy' % ('OLS/Results_OLS/OLS_'+model+'/IHDP.eval_test', time_now), eval_test)

    '''Plot results'''
    experiments = range(1,n_experiments+1)

    create_plots(eval_train, experiments, time_now, 'train', model)
    create_plots(eval_test, experiments, time_now, 'test', model)

    return eval_train, eval_test

        