This directory contains code to build models for cognition trajectory prediction. The model takes as input
features during baseline measurements of various imaging and clinical variables to predict how the
cognition of an individual will change over time. For further details, please refer to Saboo et al,
'Predicting Longitudinal Cognition Trajectory using Baseline Imaging and Clinical Variables', ISBI 2020.

Folder details:
data/raw/Example_data.csv: Provide synthetic data file (not cross-validation splits). Synthetic data is the same as used in the above manuscript.

notebooks/Example.ipynb: Example code for preprocessing data,  and training and evaluating trajectory prediction model

src/:
data/make_dataset.py: Functions to support loading and preprocessing of data
models/cognetmodel.py: Class details of the trajectory prediction model
models/interpret.py: Formatting the results of the model
visualization/visualize.py: Visualize results


The notebook contains an example of how to specify the model architecture, train model, and evaluate it. 