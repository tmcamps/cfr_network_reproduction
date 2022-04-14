import torch
import numpy as np

#%% IBM FUNCTIONS
def mmd2_lin(X,t,p):
    ''' Linear MMD '''
    # Works the same as tensorflow one --> Finished
    it = np.where(t > 0)[0]  # Find which indices have t = 1, treatment group
    ic = np.where(t < 1)[0]  # Find which indices have t = 0, control group

    X = X.clone().detach()
    Xt = X[it, :]  # Gather slices of X that are treatment
    Xc = X[ic, :] # Gather slices of X that are control

    mean_control = torch.mean(Xc,0)
    mean_treated = torch.mean(Xt,0)

    mmd = torch.sum(torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    # Works the same as tensorflow one --> Finished
    it = np.where(t > 0)[0]  # Find which indices have t = 1, treatment group
    ic = np.where(t < 1)[0]  # Find which indices have t = 0, control group

    X = X.clone().detach()
    Xt = X[it, :]  # Gather slices of X that are treatment
    Xc = X[ic, :]  # Gather slices of X that are control

    nt = Xt.shape[0]
    nc = Xc.shape[0]

    ''' Compute Euclidean distance matrix'''
    if sq:
        M = pdist2sq(Xt, Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt, Xc))

    '''Estimate lambda and delta'''
    M_mean = torch.mean(M)
    # p_drop = torch.nn.Dropout(10 / (nc * nt))
    # M_drop = p_drop(M)
    delta = (torch.max(M)).detach()
    eff_lam = (lam / M_mean).detach()

    ''' Compute new distance matrix '''
    #Mt = M
    row = delta * torch.ones((M[0:1, :]).size())
    col = torch.cat((delta * torch.ones((M[:, 0:1]).size()), torch.zeros(1, 1)),0)
    Mt = torch.cat((M, row),0)
    Mt = torch.cat((Mt, col),1)

    ''' Compute marginal vectors '''
    a = torch.cat((p * torch.ones((np.where(t > 0)[1]).shape[0],1) / nt, (1 - p) * torch.ones((1,1))),0)
    b = torch.cat(((1 - p) * torch.ones((np.where(t < 1)[1]).shape[0], 1) / nc, p * torch.ones((1, 1))), 0)

    '''Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = torch.exp(-Mlam) + 1e-6 #Added constant to avoid NaN
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / (torch.matmul(ainvK, (b / torch.transpose(torch.matmul(torch.transpose(u,0,1),K),0,1))))
    v = b / (torch.transpose(torch.matmul(torch.transpose(u,0,1), K),0,1))
    '''Calculate optimal transport matrix T*'''

    T = u * (torch.transpose(v,0,1) * K)

    if not backpropT:
        T = (T).detach()

    E = T * Mt
    wass = 2 * torch.sum(E)

    return wass

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*torch.matmul(X,torch.transpose(Y,0,1))
    nx = torch.sum(torch.square(X),1,keepdim=True)
    ny = torch.sum(torch.square(Y),1,keepdim=True)
    D = (C + torch.transpose(ny,0,1)) + nx
    return D

#%% help functions
def safe_sqrt(x):
    return torch.sqrt(torch.clamp(x, min=1e-10, max=np.inf))

def validation_split(data, val_fraction=0.3):
    """ Construct a train/validation split """
    n = data['x'].shape[0]

    n_valid = int(val_fraction*n)
    n_train = n-n_valid
    I = np.random.permutation(range(0,n))
    I_train = I[:n_train]
    I_valid = I[n_train:]

    return I_train, I_valid

def batches(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def data_per_experiment(data_train, data_test, i_exp):
    '''select train_data for experiment'''
    data_exp = {}
    data_exp['x'] = data_train['x'][:, :, i_exp - 1]
    data_exp['t'] = data_train['t'][:, i_exp - 1:i_exp]
    data_exp['yf'] = data_train['yf'][:, i_exp - 1:i_exp]
    data_exp['ycf'] = data_train['ycf'][:, i_exp - 1:i_exp]

    data_test_exp = {}
    data_test_exp['x'] = data_test['x'][:, :, i_exp - 1]
    data_test_exp['t'] = data_test['t'][:, i_exp - 1:i_exp]
    data_test_exp['yf'] = data_test['yf'][:, i_exp - 1:i_exp]
    data_test_exp['ycf'] = data_test['ycf'][:, i_exp - 1:i_exp]

    return data_exp, data_test_exp

def create_data_dictionaries(D_exp, I_train, I_valid, D_exp_test):
    d_train_f = {}
    if D_exp is not None and I_train is not None:
        d_train_f['x'] = D_exp['x'][I_train,:]
        d_train_f['t'] = D_exp['t'][I_train, :]
        d_train_f['yf'] = D_exp['yf'][I_train, :]

    d_validation_f = {}
    if D_exp is not None and I_valid is not None:

        d_validation_f['x'] = D_exp['x'][I_valid,:]
        d_validation_f['t'] = D_exp['t'][I_valid, :]
        d_validation_f['yf'] = D_exp['yf'][I_valid, :]

    d_train_cf = {}
    if D_exp is not None and I_train is not None:
        d_train_cf['x'] = D_exp['x'][I_train, :]
        d_train_cf['t'] = 1-D_exp['t'][I_train, :]
        d_train_cf['ycf'] = D_exp['ycf'][I_train, :]

    d_test_f = {}
    if D_exp_test is not None:
        d_test_f = D_exp_test

    d_test_cf = {}
    if D_exp_test is not None:
        d_test_cf['x'] = D_exp_test['x']
        d_test_cf['t'] = 1-D_exp_test['t']
        d_test_cf['ycf'] = D_exp_test['ycf']

    return d_train_f, d_validation_f, d_train_cf, d_test_f, d_test_cf

