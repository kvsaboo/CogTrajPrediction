**Overview**

This directory contains code to build a deep learning model for 5-year future cognition trajectory prediction. The model takes as input features baseline measurements of various imaging and clinical variables to predict how the cognition of an individual will change over time. For further details, please refer to [Saboo et al,
'Predicting Longitudinal Cognition Trajectory using Baseline Imaging and Clinical Variables', ISBI 2020](https://ieeexplore.ieee.org/abstract/document/9098511).

**Example**

For an example, run the [notebooks/Example.ipynb](https://github.com/kvsaboo/CogTrajPrediction/blob/master/notebooks/Example.ipynb). The notebook explains how to specify the deep learning model architecture, train model, and evaluate it.

**Directory structure**

```
.
├── data
│   └── raw
│      └── Example_data.csv 	# Provide synthetic data file (not cross-validation splits). Synthetic data is the same as used in the above manuscript.
├── notebooks
│   └── Example.ipynb 	# Example code for preprocessing data,  and training and evaluating trajectory prediction model
├── src
│   ├── data
│   │   └── make_dataset.py 	# Functions to support loading and preprocessing of data
│   ├── models
│   │   ├── cognetmodel.py 	# Class details of the trajectory prediction model
│   │   └── interpret.py 	# Formatting the results of the model
│   └── visualization
│       └── visualize.py 	# Visualize results
├── LICENSE
└── README.md 
```

**Packages**

torch - 1.7.1  
pandas - 1.1.3  
numpy - 1.19.2  
matplotlib - 3.3.2  
sklearn - 0.23.2  
pickle - 4.0  
scipy - 1.5.2  
seaborn - 0.11.0  

**Contact**

In case of any queries or comments, please contact ksaboo2@illinois.edu.


