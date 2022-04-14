import numpy as np

def pdist2(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*X.dot(Y.T)
    nx = np.sum(np.square(X),1,keepdims=True)
    ny = np.sum(np.square(Y),1,keepdims=True)
    D = (C + ny.T) + nx

    return np.sqrt(D + 1e-8)

def cf_nn(x, t):

    It = np.array(np.where(t==1))[0,:]
    Ic = np.array(np.where(t==0))[0,:]

    x_c = x[Ic,:]
    x_t = x[It,:]

    D = pdist2(x_c, x_t)

    nn_t = Ic[np.argmin(D,0)]
    nn_c = It[np.argmin(D,1)]

    return nn_t, nn_c

def pehe_nn(yf_p, ycf_p, y, x, t, nn_t=None, nn_c=None):
    if nn_t is None or nn_c is None:
        nn_t, nn_c = cf_nn(x,t)

    It = np.array(np.where(t==1))[0,:]
    Ic = np.array(np.where(t==0))[0,:]

    # Construct Counterfactual Treatment group prediction using respective NN from factual outcome
    ycf_t = 1.0*y[nn_t]

    # Difference in effect for treatment group when using counterfactual treatment group computed using NN
    eff_nn_t = ycf_t - 1.0*y[It]

    # Difference in effect for treatment group
    eff_pred_t = ycf_p[It] - yf_p[It]

    eff_pred = eff_pred_t
    eff_nn = eff_nn_t

    pehe_nn = np.sqrt(np.mean(np.square(eff_pred - eff_nn)))

    return pehe_nn


def evaluate_cont_ate(predictions, data, i_rep, i_out,nn_t, nn_c, validation_indices,validation=False, compute_policy_curve=False):

    '''Set variables from data set'''
    if validation:
        yf_p = predictions[i_rep][i_out][validation_indices[i_rep], :][:, 0].detach().numpy()
        ycf_p = predictions[i_rep][i_out][validation_indices[i_rep], :][:, 1].detach().numpy()

        x = data['x'][:, :, i_rep][validation_indices[i_rep]]  # Feature variables of 1 experiment [N,D]
        t = data['t'][:, i_rep][validation_indices[i_rep]]  # Treatment variables of 1 experiment (0 or 1)
        yf = data['yf'][:, i_rep][validation_indices[i_rep]]  # True Factual Outcomes of 1 experiment
        ycf = data['ycf'][:, i_rep][validation_indices[i_rep]]  # True Counter Factual Outcomes of 1 experiment
        mu0 = data['mu0'][:, i_rep][validation_indices[i_rep]]  # Noiseless underlying outcome distribution (control?) of 1 exp
        mu1 = data['mu1'][:, i_rep][validation_indices[i_rep]] # Noiseless underlying outcome distribution (treatment?) of 1 exp

    else:
        yf_p = predictions[i_rep][i_out][:, 0].detach().numpy()  # Prediction Factual outcomes of 1 experiment & 1 output time
        ycf_p = predictions[i_rep][i_out][:, 1].detach().numpy()  # Prediction Counter Factual outcomes of 1 exp & 1 output time

        x = data['x'][:, :, i_rep]  # Feature variables of 1 experiment [N,D]
        t = data['t'][:, i_rep]  # Treatment variables of 1 experiment (0 or 1)
        yf = data['yf'][:, i_rep]  # True Factual Outcomes of 1 experiment
        ycf = data['ycf'][:, i_rep]  # True Counter Factual Outcomes of 1 experiment
        mu0 = data['mu0'][:, i_rep]  # Noiseless underlying outcome distribution (control?) of 1 exp
        mu1 = data['mu1'][:, i_rep]  # Noiseless underlying outcome distribution (treatment?) of 1 exp

    '''True difference in effect drawn from noiseless underlying outcome distributions µ1 and µ0'''
    effect = mu1 - mu0

    '''Compute RMSE (=Root Mean Squared Error) of Factual data and Counter Factual data
    = std of residuals (prediction errors = predicted - true)'''
    rmse_f = np.sqrt(np.mean(np.square(yf_p - yf)))
    rmse_cf = np.sqrt(np.mean(np.square(ycf_p - ycf)))

    '''Predicted difference in effect; treatment - control group, 
    use counterfactual prediction for treatment and then inverse on the indices of the treatment group
    Dit snap ik niet helemaal??'''
    effect_pred = ycf_p - yf_p
    effect_pred[t > 0] = -effect_pred[t > 0]

    '''Individual Treatment Effect Prediction;  '''

    ''''Bias ATE prediction'''
    ate_pred = np.mean(effect_pred)
    bias_ate = abs(ate_pred - np.mean(effect))

    '''Estimated PEHE loss calculation'''
    pehe = np.sqrt(np.mean(np.square(effect_pred - effect)))

    pehe_appr = pehe_nn(yf_p, ycf_p, yf, x, t, nn_t, nn_c)

    return {'ate_pred': ate_pred, 'bias_ate': bias_ate,
            'rmse_fact': rmse_f, 'rmse_cfact': rmse_cf,
            'pehe': pehe, 'pehe_nn': pehe_appr}


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

            #Compute indices? of NN of the counterfactual treatment group?
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
