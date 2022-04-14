from evaluation_util import *

#%%
def evaluate_result(predictions, data, validation_indices, n_experiments, validation=False):
    n_outputs = len(predictions[0])
    n_reps = n_experiments
    eval_results = []

    '''Loop over output times = n_outputs, per experiment the outcomes are 15 times computes
    (with epochs = 600, total 3000 iterations) '''
    for i_out in range(n_outputs):
        eval_results_out = []

        '''Loop over repeated experiments = n_reps'''
        for i_rep in range(n_reps):

            '''Determine Nearest Neighbours (NN) matched on X using the euclidean distance
            this is used to calculate the PEHE-NN which estimates treatment effect of a given sample by 
            substituting true CF outcome with outcome F from respective NN'''

            if validation:
                nn_t, nn_c = cf_nn(data['x'][:, :, i_rep][validation_indices[i_rep], :],
                                   data['t'][:, i_rep][validation_indices[i_rep]])
            else:
                nn_t, nn_c = cf_nn(data['x'][:, :, i_rep], data['t'][:, i_rep])

            eval_result = evaluate_cont_ate(predictions, data, i_rep, i_out, nn_t, nn_c, validation_indices,
                                            validation, compute_policy_curve=False)

            eval_results_out.append(eval_result)

        eval_results.append(eval_results_out)

    # Reformat into dict
    eval_dict = {}
    keys = eval_results[0][0].keys()
    for k in keys:
        arr = [[eval_results[i][j][k] for i in range(n_outputs)] for j in range(n_reps)]
        v = np.array([[eval_results[i][j][k] for i in range(n_outputs)] for j in range(n_reps)])
        eval_dict[k] = v

    return eval_dict

#%%
def create_evaluation(result, data_train, data_test, configs):
    validation_indices = result['I_valid']
    n_experiments = configs['n_experiments']

    eval_train = evaluate_result(result['train'], data_train,validation_indices, n_experiments, validation=False)

    eval_valid = evaluate_result(result['train'], data_train, validation_indices, n_experiments, validation=True)

    if data_test is not None:
        eval_test = evaluate_result(result['test'], data_test, validation_indices, n_experiments,validation=False)
    else:
        eval_test = None

    return eval_train, eval_valid, eval_test

def compute_mean_std(eval):
    # Average per experiment
    ATE_avg_experiment = eval['bias_ate'][:,-1]
    pehe_avg_experiment = eval['pehe'][:,-1]
    # Averaged over all experiments
    ATE_mean = np.mean(ATE_avg_experiment)
    pehe_mean = np.mean(pehe_avg_experiment)

    # Standard Deviation over all experiments
    ATE_std = np.std(ATE_avg_experiment)
    pehe_std = np.std(pehe_avg_experiment)

    return ATE_mean, ATE_std, pehe_mean, pehe_std


def create_plots(eval_train, eval_test, configs, time_now):
    experiments = range(1, configs['n_experiments']+1)
    label = configs['label']
    # Plots
    plt.figure(figsize=(15, 4))

    # Plot ATE error
    ATE_error_train = eval_train['bias_ate']
    ATE_error_train = ATE_error_train[:,-1]
    ATE_error_test = eval_test['bias_ate']
    ATE_error_test = ATE_error_test[:,-1]


    plt.subplot(121)
    plt.plot(experiments, ATE_error_train, c = 'r', label = 'Absolute ATE error within-sample')
    plt.plot(experiments, ATE_error_test, c = 'b',  label = 'Absolute ATE error out-of-sample')
    plt.title('ATE absolute error experiment 1')
    plt.ylabel('Absolute error in ATE')
    plt.xlabel('Iterations')
    plt.legend(loc="upper left")

    # Plot PEHE error
    pehe_train = eval_train['pehe']
    pehe_train = pehe_train[:,-1]
    pehe_test = eval_test['pehe']
    pehe_test = pehe_test[:,-1]

    plt.subplot(122)
    plt.plot(experiments, pehe_train, c='r', label='PEHE error within-sample')
    plt.plot(experiments, pehe_test, c='b', label = 'PEHE error out-of-sample')
    plt.plot()
    plt.title('PEHE error experiment 1' )
    plt.ylabel('expected PEHE')
    plt.xlabel('Iterations')
    plt.legend(loc="upper left")

    plt.savefig('Results/'+label+'_plot_%s.png' % (time_now))
    plt.show()



