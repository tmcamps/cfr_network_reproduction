#Counterfactual Regression (CFR) nueral network reconstruction 

This code aims to reconstruct the counterfactual regression neural network by F. Johansson, U. Shalit and D. Sontag: 
https://arxiv.org/abs/1606.03976. The network is implemented using Pytorch Library and tested on the IHDP dataset. Testing
has been done on only 20 experiment due to limited memory space. The file run_cfr_net.py can be used to load the date, 
run the network, compute the evaluation measurements and plot the results. The network is compatible with the ipm terms: 
'wass2' and 'mmd_lin' or 'None' (=TARnet). It aims to reconstruct the results from Table 1 (https://arxiv.org/abs/1606.03976).
As comparison the IHDP data has also runned on OLS/LR-1 and OLS/LR-2 network, which can be found in the map: 'OLS'. 

