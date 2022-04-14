from Trainer import *

'''Load train and test sets'''
data_train = load_data("Data/ihdp_npci_1-1000.train.npz")
data_test = load_data("Data/ihdp_npci_1-1000.test.npz")

'''Set some hyperparameters'''
epochs = 600
b_size = 100
n_experiments = 20
model_l1 = 'l1'
model_l2 = 'l2'

'''Run model and plot results'''
eval_train, eval_test = run(model_l2, n_experiments, epochs, b_size, data_train, data_test)

ATE_mean_train, ATE_std_train, pehe_mean_train, pehe_std_train = compute_mean_std(eval_train)
ATE_mean_test, ATE_std_test, pehe_mean_test, pehe_std_test = compute_mean_std(eval_test)

print('ATE_mean_train: ', ATE_mean_train)
print('ATE_std_train: ', ATE_std_train)
print('pehe_mean_train: ', pehe_mean_train)
print('pehe_std_train: ', pehe_std_train)
print('""""""""')
print('ATE_mean_test: ', ATE_mean_test)
print('ATE_std_test: ', ATE_std_test)
print('pehe_mean_test: ', pehe_mean_test)
print('pehe_std_test: ', pehe_std_test)