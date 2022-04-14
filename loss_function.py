from util import *

def total_loss(p_alpha, t, y, y_pred, h_output_rep, configs):
    ''' Compute treatment probability (=u in paper)'''
    p_treated = np.mean(t)

    '''Compute reweighting to compensate for the difference in treatment group size in our sample (=w in paper) '''
    sample_weight = (t / 2 * p_treated) + ((1 - t) / (2 * 1 - p_treated))
    sample_weight = torch.Tensor(sample_weight)

    ''' Compute loss function, here the mean squared loss is used'''
    y = torch.Tensor(y)
    mean_squared_loss = torch.mean(sample_weight * torch.square(y - y_pred))

    ''' Compute Imbalance error with the IPM regularization term'''
    ipm_term = configs['ipm_term']
    p_ipm = 0.5

    if ipm_term == 'mm2_lin':
        imb_dist = mmd2_lin(h_output_rep, t, p_ipm)

    if ipm_term == 'wass2':
        imb_dist = wasserstein(h_output_rep, t, p_ipm, lam=10, its=10, sq=False, backpropT=True)

    if ipm_term == 'None':
        imb_dist = 0

    imb_error = p_alpha*imb_dist

    ''' Compute the prediction error '''
    pred_error = torch.sqrt(torch.mean(torch.square(y - y_pred)))

    '''Compute the total loss'''
    total_loss = mean_squared_loss + imb_error

    return total_loss, pred_error

