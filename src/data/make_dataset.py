import pandas as pd
import numpy as np
import torch


# normalize the input data array based on the normalization method and parameters of normalization
# passed into the function. If parameters are not passed, then compute then from the input data.
# INPUTS:
# d_array : input data
# normalization : string options 'none', 'positive', 'zero-one', 'standard', 'standard+zero-one', 'standard+positive'
# norm_params_dict : parameters for normalization depending on the type of normalization method.
def normalize_func(d_array, normalization, norm_params_dict='none'):

    normalization = normalization.lower()
    out_array = d_array.copy()
    
    if norm_params_dict is 'none': # if normalization parameters are not given, then calculate them from d_array
        norm_params_dict = []

        if normalization == 'none': # no normalization
            out_array = d_array
            norm_params_dict = {'ntype':normalization}

        elif normalization == 'positive': # make all values > 0
            out_array = d_array - min(d_array)
            norm_params_dict = {'ntype':normalization, 'min':min(d_array)}

        elif normalization == 'zero-one': # all values in [0,1]
            out_array = (d_array - min(d_array))/(max(d_array) - min(d_array))
            norm_params_dict = {'ntype':normalization, 'min':min(d_array), 'max':max(d_array)}

        elif normalization == 'standard': # standardize (zscore)
            out_array = (d_array - np.mean(d_array))/(np.std(d_array))
            norm_params_dict = {'ntype':normalization, 'mean':np.mean(d_array), 'std':np.std(d_array)}

        elif normalization == 'standard+zero-one': # standardize (zscore)
            tmp_array = (d_array - np.mean(d_array))/(np.std(d_array))
            out_array = (tmp_array - min(tmp_array))/(max(tmp_array) - min(tmp_array))      
            norm_params_dict = {'ntype':normalization, 'mean':np.mean(d_array), 'std':np.std(d_array),
                               'min':min(tmp_array), 'max':max(tmp_array)}

        elif normalization == 'standard+positive':
            tmp_array = (d_array - np.mean(d_array))/(np.std(d_array))
            out_array = tmp_array - min(tmp_array)
            norm_params_dict = {'ntype':normalization, 'mean':np.mean(d_array), 'std':np.std(d_array),
                               'min':min(tmp_array)}

        else:
            print("normalization technique not found")
    
    # when parameters for normalization are given in the input, use them
    # also put checks to ensure that the min and max value contraints are met
    else:
        if normalization == 'none': # no normalization
            out_array = d_array

        elif normalization == 'positive': # make all values > 0
            out_array = d_array - norm_params_dict['min']
            out_array[out_array < 0] = 0
            
        elif normalization == 'zero-one': # all values in [0,1]
            out_array = (d_array - norm_params_dict['min'])/(norm_params_dict['max'] - norm_params_dict['min'])
            out_array[out_array < 0] = 0
            out_array[out_array > 1] = 1
            
        elif normalization == 'standard': # standardize (zscore)
            out_array = (d_array - norm_params_dict['mean'])/(norm_params_dict['std'])
            
        elif normalization == 'standard+zero-one': # standardize (zscore)
            tmp_array = (d_array - norm_params_dict['mean'])/(norm_params_dict['std'])
            out_array = (tmp_array - norm_params_dict['min'])/(norm_params_dict['max'] - norm_params_dict['min'])
            out_array[out_array < 0] = 0
            out_array[out_array > 1] = 1
            
        elif normalization == 'standard+positive':
            tmp_array = (d_array - norm_params_dict['mean'])/(norm_params_dict['std'])
            out_array = tmp_array - norm_params_dict['min']
            out_array[out_array < 0] = 0
            
        else:
            print("normalization technique and parameters not found")
        
    return out_array, norm_params_dict


# split the input dataframe into training, validatin, and test sets and normalize the data using parameters
# computed from the training data
# data: dataframe of data
# train_frac: fraction of data used for training
# test_frac: fraction of data used for testing. train_frac+test_frac <= 1. Sampling is done without replacement.
# Xvar: list of variables in the dataframe that become features
# Yvar: list of variables in the dataframe that become labels
# normalization: string mentioning normalization technique
# randomseed : seed to control the splitting of the data
def train_val_test_split_wnorm(data, train_frac, test_frac, Xvar, Yvar, normalization, randomseed=24601):
    
    # split data into training, test and validation sets. It will be a dataframe
    data_copy = data.copy()
    num_train_samples = round(len(data_copy)*train_frac)
    num_val_samples = round(len(data_copy)*(1-train_frac-test_frac))
    
    permrows = np.random.RandomState(seed=randomseed).permutation(np.arange(0,len(data_copy)))

    train_data = data_copy.iloc[permrows[:num_train_samples]]
    val_data = data_copy.iloc[permrows[num_train_samples:(num_train_samples+num_val_samples)]]
    test_data = data_copy.iloc[permrows[(num_train_samples+num_val_samples):]]

    # normalized data frames
    train_data_norm = train_data.copy()
    val_data_norm = val_data.copy()
    test_data_norm = test_data.copy()
    
    normalize_cols = list(normalization.keys())
    normparam_dict = dict.fromkeys(normalize_cols)

    for fieldname in normalize_cols:
        train_data_norm[fieldname], normparam_dict[fieldname] = normalize_func(train_data[fieldname], normalization[fieldname])
        val_data_norm[fieldname] = normalize_func(val_data[fieldname], normalization[fieldname], normparam_dict[fieldname])[0] # the [0] index is important because function returns multiple output
        test_data_norm[fieldname] = normalize_func(test_data[fieldname], normalization[fieldname], normparam_dict[fieldname])[0]
    
    # split data into X and Y and convert to numpy array
    train_X = train_data_norm[Xvar]
    train_Y = train_data_norm[Yvar]
    val_X = val_data_norm[Xvar]
    val_Y = val_data_norm[Yvar]
    test_X = test_data_norm[Xvar]
    test_Y = test_data_norm[Yvar]
    
    return train_X, val_X, test_X, train_Y, val_Y, test_Y, normparam_dict


# remove rows containing NaN values in the dataframe
def preprocessing_rmnan(df, features, labels):
    # remove nan values from features
    ppdf = df.dropna(subset=features).copy()
    return ppdf

# transform the X and Y data to be in the format expected by the torch model
# data_X: features
# data_Y: labels i.e., trajectory
# br_features: list of variables in structural features (input to function r in the trajectory prediction model)
# clinical_features: list of variables in clinical features (input to function g_t in the trajectory prediction model)
# col_cog0: string corresponding to baseline cognition score variable in the dataframe (input to function h in the trajectory prediction model)
# labels: list of variables corresponding to labels or trajectory
def format_for_CogNet(data_X, data_Y, br_features, clinical_features, col_cog0, labels):
    data_x_br = torch.tensor(data_X[br_features].values, dtype=torch.float)
    data_x_cf = torch.tensor(data_X[clinical_features].values, dtype=torch.float)
    data_cog0 = torch.tensor(data_X[col_cog0].values, dtype=torch.float).view(-1,1)

    ground_truth = torch.tensor(data_Y[labels].values, dtype=torch.float, requires_grad=True)
    ground_truth_mask = np.ones(ground_truth.shape)
    ground_truth_mask[ground_truth.detach().numpy()==-10] = 0 # MAGIC NUMBER
    ground_truth_mask = torch.tensor(ground_truth_mask, dtype=torch.float)
    
    return data_x_br, data_x_cf, data_cog0, ground_truth, ground_truth_mask



# normalizing the trajectory values using parameters computed during baseline and replacing
# missing values in the trajectory with magic number -10.
def normalize_labels(data_Y, labels_to_norm, normtype, normparams, keep_invalid_entries=True):
    data_Y_copy = data_Y.copy()
    
    for fieldname in labels_to_norm:
        data_Y_copy[fieldname] = normalize_func(data_Y_copy[fieldname], normtype, normparams)[0]
        
    # if there were invald entries (data not collected) and that information
    # should be retained in the normalized_labels
    if keep_invalid_entries == True:
        data_Y_copy[data_Y == -10] = -10
        
    return data_Y_copy


def explode_df(df_to_explode):
    col_cross = [x for x in df_to_explode.columns if 'fu' not in x]
    col_cog = [x for x in df_to_explode.columns if 'fu' in x]    
    df_ls = []
    for idx,col_year in enumerate(col_cog):
        df_tmp = df_to_explode[col_cross+[col_year]].copy()
        df_tmp['delay'] = idx+1
        df_tmp.rename(columns={col_year:'fu'}, inplace=True)
        df_ls.append(df_tmp)  
    return pd.concat(df_ls)
