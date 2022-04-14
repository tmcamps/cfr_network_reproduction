from Util_OLS import *

def train_OLS_l2(data, net, optimizer, criterion, i, b_size, t):
    """
    Trains network for one epoch in batches.

    Args:
        data: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    inputs = data[0]
    X_exp = inputs[:, :, i]
    yf = data[1]
    indices = np.arange(0, X_exp.size(dim=0))
    y_preds = torch.zeros_like(data[0][:,0,0])

    # iterate through batches
    for batch in batches(indices, b_size):
        # get the inputs; data is a list of [inputs, labels]
        inputs = X_exp[batch, :]
        labels = yf[batch, i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net.forward(inputs, t[batch])
        flat_output = torch.flatten(outputs)
        loss = criterion(flat_output.float(), labels.float())
        loss.backward()
        optimizer.step()

        y_preds[batch] = flat_output

    return y_preds
        
def test_OLS_l2(data, net, criterion, i, b_size, t):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        net: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """

    y_preds = torch.zeros_like(data[0][:,0,0])

    inputs = data[0]
    X_exp = inputs[:, :, i]
    yf = data[1]
    indices = np.arange(0, X_exp.size(dim=0))
    
    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with torch.no_grad():
        # iterate through batches
        for batch in batches(indices, b_size):
            # get the inputs; data is a list of [inputs, labels]
            inputs = X_exp[batch, :]
            labels = yf[batch, i]

            # forward pass
            outputs = net.forward(inputs, t[batch])
            flat_output = torch.flatten(outputs)
            loss = criterion(flat_output.float(), labels.float())

            y_preds[batch] = flat_output

    return y_preds

####################
### Define model ###
####################

class OLSnet_l2(nn.Module):
    """
    Simple fully connected neural network with residual connections in PyTorch.
    Layers are defined in __init__ and forward pass implemented in forward.
    """
    
    def __init__(self):
        super(OLSnet_l2, self).__init__()

        self.fc1 = nn.Linear(25, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3_t = nn.Linear(200, 1)
        self.fc3_c = nn.Linear(200, 1)

    def forward(self, x, t):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        '''1. Split output layers in t=0 and t=1'''
        it = torch.where(t > 0)[0]
        ic = torch.where(t < 1)[0]

        h_t = h[it, :]    # Gather slices of h that are treatment
        h_c =  h[ic, :]   # Gather slices of h that are control

        y_pred_t = self.fc3_t(h_t)
        y_pred_c = self.fc3_c(h_c)

        '''2. Combine outcome prediction y of treatment and control group again to one outcome prediction'''
        y_pred = torch.zeros_like(t)
       
        y_pred[it] = torch.flatten(y_pred_t)
        y_pred[ic] = torch.flatten(y_pred_c)

        return y_pred
