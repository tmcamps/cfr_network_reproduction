''' Import packages for the model '''
from util import *

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



