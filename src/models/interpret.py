import numpy as np
import pandas as pd
import torch
import pickle
from scipy import stats
from copy import deepcopy
from src.data import make_dataset

# format the predicted trajectory for ease of visualization
def format_result_for_viz(prediction, gt_Y, Series_cog0):
    
    col_list = gt_Y.columns
    
    # define dataframe of prediction
    pred_list = [labval+'_pred' for labval in col_list]
    pred_df = pd.DataFrame(torch.cat(prediction, dim=1).detach().numpy(), index=gt_Y.index, columns=col_list)
    
    # replace magic number of NaN
    pred_df[gt_Y==-10] = np.NaN
    pred_df.columns = pred_list
    
    # define dataframe of ground truth
    truth_map = dict.fromkeys(col_list)
    for labval in col_list:
        truth_map[labval] = labval + '_truth'
    
    truth_df = gt_Y.rename(columns=truth_map)
    truth_df[truth_df==-10] = np.NaN
    
    # concatenate the prediction and ground truth based on patient ID as index
    pred_truth_df = pd.concat( (pred_df, truth_df), axis=1)
    
    # drop subjects for which there are no ground truth available
    pred_truth_df.dropna(subset=pred_truth_df.columns, how='all', inplace=True)

    pred_truth_df[Series_cog0.name] = Series_cog0
    
    return pred_truth_df

