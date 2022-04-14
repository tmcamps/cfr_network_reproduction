from training import *
from Model import *
from evaluation import *
import pickle
from datetime import datetime

''' Set configs for running'''
configs = {
    "D_in" : 25,
    "H_representation" : 200,
    "H_regression" : 100,
    "weight_init" : 0.1,
    "p_alpha" : 0.3,
    "p_lambda" : 1e-4,
    "n_in" : 3,
    "n_out" : 3,
    "n_experiments" : 20,
    "ipm_term" : 'wass2',
    "batch_size": 100,
    "epochs": 600,
    "label": 'CFR_net',
    "datadir":'Data/',
    "resultdir":'Results/',
    "data_train":'ihdp_npci_1-1000.train.npz',
    "data_test":'ihdp_npci_1-1000.test.npz'
}

'''Import data'''
data_train = load_data(configs['datadir']+configs['data_train'])
data_test = load_data(configs['datadir']+configs['data_test'])

#%%
''' Run cfr net with configs and gather all results of the n experiments
    Args:
    - all_preds_train: Consists of all the prediction outcomes of every 200 iterations of every experiment;
                       Factual & counterfactual outcomes; 
                       List for every experiment (n_experiments) a list with every 20 epochs an Outcome Tensor, 
                       Outcome Tensor consists of 672 patients with each a factual and counterfactual outcome.
    - all_preds_test:  Consists of all prediction outcomes of every 200 iterations of every experiment; 
                       Factual & counterfactual outcomes; 
                       List for every experiment (n_experiments) a list with every 40 epochs an Outcome Tensor, 
                       Outcome Tensor consists of 672 patients with each a factual and counterfactual outcome
    - all_obj_losses: consists of all objective losses of all iterations (batches * epochs) and all experiments
    - all_I_valid: consists of all validation set indices used per experiment 
'''
all_preds_train, all_preds_test, all_obj_losses, all_I_valid = run(data_train, data_test, configs)

''' Save outcomes as dictionary and save results'''
result = {'train':all_preds_train, "test":all_preds_test, "all_obj_losses": all_obj_losses, "I_valid":all_I_valid}

file_path = configs['resultdir'] + configs['label'] + '_' + configs['ipm_term'] + 'pkl'

a_file = open(file_path, "wb")
pickle.dump(result, a_file)
a_file.close()

#%% Import data for evaluation
file_path = configs['resultdir'] + configs['label'] + '_' + configs['ipm_term'] + '.pkl'

#%%
a_file = open("Results/CFR_net_Nonepkl", "rb")
result = pickle.load(a_file)

#%%
''' Create evaluation dictionaries '''
eval_train, eval_valid, eval_test = create_evaluation(result, data_train, data_test, configs)

''' Plot evaluations'''
time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

#%%
create_plots(eval_train, eval_test, configs, time_now)

#%%
'''Compute averages and std within sample (ws) & out of sample (oos)'''
ATE_mean_ws, ATE_std_ws, pehe_mean_ws, pehe_std_ws = compute_mean_std(eval_train)
ATE_mean_oos, ATE_std_oos, pehe_mean_oos, pehe_std_oos = compute_mean_std(eval_test)



