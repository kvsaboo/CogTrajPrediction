import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import mean_squared_error
import numpy as np


## mean squared error loss function 
class modifiedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, prediction, ground_truth, gt_mask):
        diffval = torch.mul((torch.cat(prediction, dim=1) - ground_truth).pow(2), gt_mask)
        lossval = diffval.sum()/gt_mask.sum()
        return lossval

# Average mean squared error first across individuals, then across years     
class modifiedMSELoss_new(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, ground_truth, gt_mask):
        diffval = torch.mul((torch.cat(prediction, dim=1) - ground_truth).pow(2), gt_mask)
        lossval = (diffval.sum(dim=0)/gt_mask.sum(dim=0)).mean()
        return lossval


def mseloss_all(prediction, ground_truth, gt_mask=None):
    if len(ground_truth.shape) > 1:
        num_vars = ground_truth.shape[1]
    else:
        num_vars = 0
        
    mselossarr = np.zeros(num_vars+1) # one for each variable and one overall
    
    for varnum in range(len(mselossarr)-1):
        if gt_mask is not None:
            mselossarr[varnum] = mseloss(prediction[:,varnum], ground_truth[:,varnum], gt_mask[:,varnum])
        else:
            mselossarr[varnum] = mseloss(prediction[:,varnum], ground_truth[:,varnum])
    
    mselossarr[-1] = mseloss(prediction, ground_truth, gt_mask)
    
    return mselossarr



def mseloss(prediction, ground_truth, gt_mask=None):
    if type(prediction) == torch.Tensor:
        prediction = prediction.detach().numpy()
        
    if type(ground_truth) == torch.Tensor:
        ground_truth = ground_truth.detach().numpy()
    
    # define mask if it isn't already defined
    if type(gt_mask) != torch.Tensor:
        gt_mask = np.ones(ground_truth.shape, dtype=bool)
        gt_mask[ground_truth == -10] = 0
    else:
        gt_mask = np.asarray(gt_mask, dtype=bool)
        
    error = mean_squared_error(prediction[gt_mask], ground_truth[gt_mask])
    return error



# model with dropout in functions g, h
class CogNetWtShare_drop_leaky_2_2(nn.Module):
    def __init__(self, col_br, col_cf, num_br_hid, num_br_hid_2, num_br_long, num_br_long_2):
        super().__init__()
        self.drop = nn.Dropout() # default dropout of 0.5
        
        self.hidden_br = nn.Linear(len(col_br), num_br_hid)
        self.hidden_br2 = nn.Linear(num_br_hid, num_br_hid_2)

        self.hidden_cog1 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog1 = nn.Linear(num_br_long, num_br_long_2)

        self.hidden_cog2 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog2 = nn.Linear(num_br_long, num_br_long_2)

        self.hidden_cog3 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog3 = nn.Linear(num_br_long, num_br_long_2)

        self.hidden_cog4 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog4 = nn.Linear(num_br_long, num_br_long_2)

        self.hidden_cog5 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog5 = nn.Linear(num_br_long, num_br_long_2)

        self.output_cog = nn.Linear(num_br_long_2+1, 1)
        
    def forward(self, x_br, x_cf, cog0):
        x_br = F.relu(self.hidden_br(self.drop(x_br)))
        x_br = F.relu(self.hidden_br2(self.drop(x_br)))

        x_hidden_cog1 = self.concat_hidden(x_br, x_cf, 0)
        x_hidden_cog1 = F.leaky_relu(self.hidden_cog1(x_hidden_cog1), negative_slope=0.1)
        x_hidden_cog1 = F.leaky_relu(self.hidden_2_cog1(x_hidden_cog1), negative_slope=0.1)
        x_output_cog1 = torch.cat((x_hidden_cog1, cog0), dim=1)
        x_output_cog1 = self.output_cog(x_output_cog1)
        
        x_hidden_cog2 = self.concat_hidden(x_br, x_cf, 1)
        x_hidden_cog2 = F.leaky_relu(self.hidden_cog2(x_hidden_cog2), negative_slope=0.1)
        x_hidden_cog2 = F.leaky_relu(self.hidden_2_cog2(x_hidden_cog2), negative_slope=0.1)
        x_output_cog2 = torch.cat((x_hidden_cog2, x_output_cog1), dim=1)
        x_output_cog2 = self.output_cog(x_output_cog2)
        
        x_hidden_cog3 = self.concat_hidden(x_br, x_cf, 2)
        x_hidden_cog3 = F.leaky_relu(self.hidden_cog3(x_hidden_cog3), negative_slope=0.1)
        x_hidden_cog3 = F.leaky_relu(self.hidden_2_cog3(x_hidden_cog3), negative_slope=0.1)
        x_output_cog3 = torch.cat((x_hidden_cog3, x_output_cog2), dim=1)
        x_output_cog3 = self.output_cog(x_output_cog3)        
        
        x_hidden_cog4 = self.concat_hidden(x_br, x_cf, 3)
        x_hidden_cog4 = F.leaky_relu(self.hidden_cog4(x_hidden_cog4), negative_slope=0.1)
        x_hidden_cog4 = F.leaky_relu(self.hidden_2_cog4(x_hidden_cog4), negative_slope=0.1)
        x_output_cog4 = torch.cat((x_hidden_cog4, x_output_cog3), dim=1)
        x_output_cog4 = self.output_cog(x_output_cog4) 
        
        x_hidden_cog5 = self.concat_hidden(x_br, x_cf, 4)
        x_hidden_cog5 = F.leaky_relu(self.hidden_cog5(x_hidden_cog5), negative_slope=0.1)
        x_hidden_cog5 = F.leaky_relu(self.hidden_2_cog5(x_hidden_cog5), negative_slope=0.1)
        x_output_cog5 = torch.cat((x_hidden_cog5, x_output_cog4), dim=1)
        x_output_cog5 = self.output_cog(x_output_cog5)

        return x_output_cog1,x_output_cog2,x_output_cog3,x_output_cog4,x_output_cog5,x_hidden_cog1,x_hidden_cog2,x_hidden_cog3,x_hidden_cog4,x_hidden_cog5,x_br
    
    def concat_hidden(self, ts1, ts2, delay):
        N = ts1.shape[0]
        return torch.cat((ts1,ts2,torch.tensor(np.ones(N)*delay, dtype=torch.float).view(-1,1)), dim=1)


# model without dropout in all functions (r,g,h)
class CogNetWtShare_dropall_leaky_2_2(nn.Module):
    def __init__(self, col_br, col_cf, num_br_hid, num_br_hid_2, num_br_long, num_br_long_2):
        super().__init__()
        self.drop = nn.Dropout() # default dropout of 0.5
        
        self.hidden_br = nn.Linear(len(col_br), num_br_hid)
        self.hidden_br2 = nn.Linear(num_br_hid, num_br_hid_2)

        self.hidden_cog1 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog1 = nn.Linear(num_br_long, num_br_long_2)

        self.hidden_cog2 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog2 = nn.Linear(num_br_long, num_br_long_2)

        self.hidden_cog3 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog3 = nn.Linear(num_br_long, num_br_long_2)

        self.hidden_cog4 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog4 = nn.Linear(num_br_long, num_br_long_2)

        self.hidden_cog5 = nn.Linear(num_br_hid_2+len(col_cf)+1, num_br_long)
        self.hidden_2_cog5 = nn.Linear(num_br_long, num_br_long_2)

        self.output_final_cog = nn.Linear(num_br_long_2+1, 1)
        
    def forward(self, x_br, x_cf, cog0):
        x_br = F.relu(self.hidden_br(self.drop(x_br)))
        x_br = F.relu(self.hidden_br2(self.drop(x_br)))

        x_hidden_cog1 = self.concat_hidden(x_br, x_cf, 0)
        x_hidden_cog1 = F.leaky_relu(self.hidden_cog1(self.drop(x_hidden_cog1)), negative_slope=0.1)
        x_hidden_cog1 = F.leaky_relu(self.hidden_2_cog1(self.drop(x_hidden_cog1)), negative_slope=0.1)
        x_output_cog1 = torch.cat((x_hidden_cog1, cog0), dim=1)
        x_output_cog1 = self.output_final_cog(x_output_cog1)
        
        x_hidden_cog2 = self.concat_hidden(x_br, x_cf, 1)
        x_hidden_cog2 = F.leaky_relu(self.hidden_cog2(self.drop(x_hidden_cog2)), negative_slope=0.1)
        x_hidden_cog2 = F.leaky_relu(self.hidden_2_cog2(self.drop(x_hidden_cog2)), negative_slope=0.1)
        x_output_cog2 = torch.cat((x_hidden_cog2, x_output_cog1), dim=1)
        x_output_cog2 = self.output_final_cog(x_output_cog2)
        
        x_hidden_cog3 = self.concat_hidden(x_br, x_cf, 2)
        x_hidden_cog3 = F.leaky_relu(self.hidden_cog3(self.drop(x_hidden_cog3)), negative_slope=0.1)
        x_hidden_cog3 = F.leaky_relu(self.hidden_2_cog3(self.drop(x_hidden_cog3)), negative_slope=0.1)
        x_output_cog3 = torch.cat((x_hidden_cog3, x_output_cog2), dim=1)
        x_output_cog3 = self.output_final_cog(x_output_cog3)        
        
        x_hidden_cog4 = self.concat_hidden(x_br, x_cf, 3)
        x_hidden_cog4 = F.leaky_relu(self.hidden_cog4(self.drop(x_hidden_cog4)), negative_slope=0.1)
        x_hidden_cog4 = F.leaky_relu(self.hidden_2_cog4(self.drop(x_hidden_cog4)), negative_slope=0.1)
        x_output_cog4 = torch.cat((x_hidden_cog4, x_output_cog3), dim=1)
        x_output_cog4 = self.output_final_cog(x_output_cog4) 
        
        x_hidden_cog5 = self.concat_hidden(x_br, x_cf, 4)
        x_hidden_cog5 = F.leaky_relu(self.hidden_cog5(self.drop(x_hidden_cog5)), negative_slope=0.1)
        x_hidden_cog5 = F.leaky_relu(self.hidden_2_cog5(self.drop(x_hidden_cog5)), negative_slope=0.1)
        x_output_cog5 = torch.cat((x_hidden_cog5, x_output_cog4), dim=1)
        x_output_cog5 = self.output_final_cog(x_output_cog5)

        return x_output_cog1,x_output_cog2,x_output_cog3,x_output_cog4,x_output_cog5,x_hidden_cog1,x_hidden_cog2,x_hidden_cog3,x_hidden_cog4,x_hidden_cog5,x_br
    
    def concat_hidden(self, ts1, ts2, delay):
        N = ts1.shape[0]
        return torch.cat((ts1,ts2,torch.tensor(np.ones(N)*delay, dtype=torch.float).view(-1,1)), dim=1)
