import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''Create batches'''
def batches(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

'''Load the data sets'''
def load_data(file_name):
    data_in = np.load(file_name)
    data = {'x': data_in['x'], 't' : data_in['t'], 'yf': data_in['yf'], 'ycf': data_in['ycf'], \
            'mu0': data_in['mu0'], 'mu1': data_in['mu1'], 'ATE': data_in['ate'], 'YMUL': data_in['ymul'], \
            'YADD': data_in['yadd']}

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]
    if 'train' in file_name:
        print('Number of features in IHDP data set:', data['dim'])
        print('Number of patients in IHDP train data set', data['n'])
        print('Number of realizations of the outcomes', data['x'].shape[2])
        print('Treatment data = data["t"], if t = 1; child gets treatment, if t = 0; child does not get treatment')
        print('Y_factual data = data["yf"]')
        print('Y_counter_factual data = data["ycf"]')

    return data

'''Plot evaluations'''
def create_plots(eval, experiments, time_now, label, ols):

    # Plots
    plt.figure(figsize=(15, 4))

    # Plot ATE error 
    ATE_error = eval['bias_ate']
    ATE_error = np.mean(ATE_error, 1)

    plt.subplot(131)
    plt.plot(experiments, ATE_error)
    plt.title('ATE absolute error ' + label +  ' set')
    plt.ylabel('Absolute error in ATE')
    plt.xlabel('Iterations')

    # Plot PEHE error
    pehe = eval['pehe']
    pehe = np.mean(pehe, 1)

    plt.subplot(132)
    plt.plot(experiments, pehe)
    plt.title('PEHE error ' + label + ' set')
    plt.ylabel('expected PEHE')
    plt.xlabel('Iterations')

    # Plot PEHE_NN error train set
    pehe_nn = eval['pehe_nn']
    pehe_nn = np.mean(pehe_nn, 1)

    plt.subplot(133)
    plt.plot(experiments, pehe_nn)
    plt.title('PEHE-NN error ' + label + ' set')
    plt.ylabel('expected PEHE-NN')
    plt.xlabel('Iterations')

    if ols == "l1":
        plt.savefig('OLS/Results_OLS/OLS_l1/'+label+'_plot_%s.png' % (time_now))
    else:
        plt.savefig('OLS/Results_OLS/OLS_l2/'+label+'_plot_%s.png' % (time_now))

    plt.show()

'''Calculate mean and std'''
def compute_mean_std(eval):
    # Average per experiment
    ATE_avg_experiment = np.mean(eval['bias_ate'],1)
    pehe_avg_experiment = np.mean(eval['pehe'], 1)
    # Averaged over all experiments
    ATE_mean = np.mean(ATE_avg_experiment)
    pehe_mean = np.mean(pehe_avg_experiment)

    # Standard Deviation over all experiments
    ATE_std = np.std(ATE_avg_experiment)
    pehe_std = np.std(pehe_avg_experiment)

    return ATE_mean, ATE_std, pehe_mean, pehe_std