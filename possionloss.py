import torch
from torch import nn

class PoissonLikelihood_loss(nn.Module):
    def __init__(self, max_val=0):
        '''
        Poisson Likelihood loss function
        Email: jchen245@jhmi.edu
        Date: 02/21/2021
        :param max_val: the maximum value of the target.
        '''
        super(PoissonLikelihood_loss, self).__init__()
        self.max_val = max_val

    def forward(self, y_pred, y_true):
        eps = 1e-6
        y_pred = y_pred.view(y_pred.size[0], -1)
        y_true = y_true.view(y_true.size[0], -1)

        p_l = -y_true+y_pred*torch.log(y_true + eps)-torch.lgamma(y_pred+1)
        return -torch.mean(p_l)
