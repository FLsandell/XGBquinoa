#!/usr/bin/env python3

# XGBquinoa
# (c) Felix Leopold Sandell, 19.9.2023
# Institute of Computational Biology
# BOKU, Vienna
# Contact: felix.sandell@boku.ac.at

# This code is intended to be used as a jupyter notebook. The corresponding notebook can be found on my GitHub: https://github.com/FLsandell/XGBquinoa/tree/main

import pandas as pd
import numpy as np
import io
import os
import sys
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
from hyperopt import STATUS_OK
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt


# Part 1 - Parameter optimization and modell fitting

# Import the 0/1/2 matrix. The matrix is importet transposed
# since pandas is faster importing files with a large number of lines than collumns

Quinoa_01 = pd.read_csv('/path/to/input/012/Matrix.txt', sep = "\t")

# Set row index
Quinoa_01 = Quinoa_01.set_index('ACC')

# Transpose the file
Quinoa_01 = Quinoa_01.T

# Keep sets with all colours. This is only needed for the all-colours LDA and PCA at the
# end of the notebook. Remove the # if you want to calculate this. Keep in mind keeping
# both tables is memory expensive.

# Quinoa_original = Quinoa_01


# Import y
y = pd.read_csv('path/to/targets/colour/file.txt', sep = "\t")


# index y to have the same index as Quinoa_01
y_index = y.set_index(Quinoa_01.index)

# combine X and y
Quinoa_01['Colour'] = y_index

# We train our models only on accessions coloured beige, orange and white, therefore
# the other accessions are removed

Quinoa_01 = Quinoa_01[Quinoa_01.Colour != "GreenishRed"]
Quinoa_01 = Quinoa_01[Quinoa_01.Colour != "light"]
Quinoa_01 = Quinoa_01[Quinoa_01.Colour != "unclear"]
Quinoa_01 = Quinoa_01[Quinoa_01.Colour != "yellow"]
Quinoa_01 = Quinoa_01[Quinoa_01.Colour != "black"]
Quinoa_01 = Quinoa_01[Quinoa_01.Colour != "red"]
Quinoa_01 = Quinoa_01[Quinoa_01.Colour != "brown"]
Quinoa_01 = Quinoa_01[Quinoa_01.Colour != "mix"]


# The parameter optimisation is calculated using hyperopt, we therefore need functions to
# extract informations from trial objects

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['Trained_Model']


def getBestLossfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['loss']


def getBestRoundsfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['rounds']


# We define the search Space for parameter optimization

space={ 'learning_rate'     : hp.choice('learning_rate', np.logspace(np.log10(0.005), np.log10(0.2), base = 10, num = 400)),
        'max_depth': hp.choice('max_depth', range(10, 32, 1)),
        'gamma': hp.choice('gamma', [0.5, 1, 1.5, 2, 3]),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'reg_alpha' : hp.uniform('reg_alpha', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5 , 0.9 ),
        'subsample' : hp.uniform('subsample', 0.5,0.9),
        'min_child_weight' : hp.quniform('min_child_weight', 4, 10, 1)
    }


# We define the function for parameter optimization

def objective(space):
    params = {"objective" : "multi:softprob", "num_class" : 3,
                    "max_depth" : int(space['max_depth']), "gamma" : space['gamma'],
                    "min_child_weight":int(space['min_child_weight']),
                    "colsample_bytree" : (space['colsample_bytree']),
                    "subsample":(space['subsample']),
                    "learning_rate" :(space['learning_rate']) ,
                     "reg_lambda" : (space['reg_lambda']),
                     "reg_alpha": (space['reg_alpha'])}

    xgboost_train = xgb.DMatrix(data=X_train, label=y_train, weight = X_weights)

    #the maximum of rounds is set to 1000, but we have early stopping at round 25

    num_boost_round=1000

    xgb_cv = xgb.cv(dtrain=xgboost_train,  params=params, num_boost_round=num_boost_round, early_stopping_rounds = 20, nfold=4,  metrics = 'auc',seed=8, stratified=True)

    n_rounds = len(xgb_cv["test-auc-mean"])
    cv_score = xgb_cv["test-auc-mean"][n_rounds-1]

    print( 'CV finished n_rounds={} cv_score={:7.5f}'.format( n_rounds, cv_score ) )


    print ("SCORE:", cv_score)
    return {'loss': -cv_score, 'status': STATUS_OK,'Trained_Model': params, 'rounds' : n_rounds }


# Set X and y
y = Quinoa_01['Colour'].values
X = Quinoa_01.drop(['Colour'], axis=1).values



# As working with a unbalanced dataset we set wheights

classes_weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=Quinoa_01['Colour'])


# Use this cell if you want to add the first 10 principal components, to correct for population
# Structure
#pca = PCA(n_components=10)
#principalComponents_large = pca.fit_transform(X)
#X = np.append(X,principalComponents_large, axis = 1)


# The lables are encoded and we split the training and the test set based on a random seed
# For the publication we used the seeds 0-99. The best performing seed seed, the split reported
# in the publication is 47. For clarity and reproducibility we also report the worst seed, which
# is 67


le = preprocessing.LabelEncoder()
label_encoder = le.fit(y)
y_encoded = label_encoder.transform(y)

seed = 67

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify = y_encoded ,random_state = seed)
X_weights, X_test_drop, y_train_drop, y_test_drop = train_test_split(classes_weights, y_encoded, stratify = y_encoded ,random_state = seed)

# Perform Hyper Parameter Optimization with 250 rounds of HyperOpt
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=250,
            trials=trials)

print (best)

# Get the best performing model from Hyperopt
best_model = getBestModelfromTrials(trials)

# Get the number of rounds, before early stopping

rounds = getBestRoundsfromTrials(trials)

# trained parameters for split 67
#best_model = {'objective': 'multi:softprob',
 #'num_class': 3,
 #'max_depth': 11,
 #'gamma': 2,
 #'min_child_weight': 6,
 #'colsample_bytree': 0.5093555761044726,
 #'subsample': 0.5450743591736262,
 #'reg_lambda': 0.7333308928373442,
 #'reg_alpha': 0.7333308928373442,
 #'learning_rate': 0.013952623219629859}



# trained parameters for split 47

best_model = {'objective': 'multi:softprob',
    'num_class': 3,
    'max_depth': 25,
    'gamma': 1.5,
    'min_child_weight': 8,
    'colsample_bytree': 0.7193577374394876,
    'subsample': 0.6481597809269125,
    'learning_rate': 0.020956759830646633}

# Test our model against the Training set

xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
clf=xgb.XGBClassifier(**best_model, n_estimators = 50, eval_metric = "auc")

# Fit the model

clf.fit(X_train, y_train, sample_weight=X_weights)

# predict targets

y_pred = clf.predict(X_test)

# Classification Report
classification = sklearn.metrics.classification_report(y_test,y_pred)

print(classification)

# Part 2 - Feature Importance


# Create a dataframe without the targets

X_df = Quinoa_01.drop(['Colour'], axis = 1)


# Create a list of all feature names

orig_feature_names = X_df.columns.values.tolist()

# Add the original feature names to the model

clf.get_booster().feature_names = orig_feature_names

# Calculate feature importances

feats = clf.get_booster().get_score(importance_type="gain")

# create a list with the feature importances

features_list = feats.keys()

# Create a dataframe with only variants that resulted in a gain of prediction accuracy

X_small = X_df.loc[:,features_list]

# Part 3 - PCAs and LDAs
# PCA

# Calculate a PCA with 2 components

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_small)

# Create a table

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf = principalDf.set_index(Quinoa_01.index)
finalDf = pd.concat([principalDf, Quinoa_01[['Colour']]], axis = 1)

# Plot the PCA on a reduced dataset


fig = plt.figure(figsize = (11,11))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA', fontsize = 20)
targets = ['orange', 'beige']
colors = ['darkorange', 'darkkhaki']
markers = ['o','P']
for target, color, marker in zip(targets,colors,markers):
    indicesToKeep = finalDf['Colour'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 40
               , marker = marker)

targets = ['white']
colors = [ 'dimgrey']
markers = ['o']
for target, color, marker in zip(targets,colors,markers):
    indicesToKeep = finalDf['Colour'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , edgecolors = color
               , s = 40
               , marker = marker
               , facecolors='none')

targets = ['Orange','Beige','White']
ax.legend(targets, prop={'size': 12})
ax.tick_params(labelsize=12)
ax.grid()

# LDA

# Calculate a LDA with 2 components
y_lda = Quinoa_01[['Colour']].values.ravel()

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_small, y_lda)


# Create a table

ldaDf = pd.DataFrame(data = X_lda
             , columns = ['lda component 1', 'lda component 2'])
ldaDf = ldaDf.set_index(Quinoa_01.index)
finalDf_lda = pd.concat([ldaDf, Quinoa_01[['Colour']]], axis = 1)

# Plot the PCA on a reduced dataset

fig = plt.figure(figsize = (11,11))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Linear Discriminant 1', fontsize = 15)
ax.set_ylabel('Linear Discriminant 2', fontsize = 15)
ax.set_title('LDA', fontsize = 20)
targets = ['orange', 'beige']
colors = ['darkorange', 'darkkhaki']
markers = ['o','P']
for target, color, marker in zip(targets,colors,markers):
    indicesToKeep = finalDf['Colour'] == target
    ax.scatter(finalDf_lda.loc[indicesToKeep, 'lda component 1']
               , finalDf_lda.loc[indicesToKeep, 'lda component 2']
               , c = color
               , s = 40
               , marker = marker)

targets = ['white']
colors = [ 'dimgrey']
markers = ['o']
for target, color, marker in zip(targets,colors,markers):
    indicesToKeep = finalDf['Colour'] == target
    ax.scatter(finalDf_lda.loc[indicesToKeep, 'lda component 1']
               , finalDf_lda.loc[indicesToKeep, 'lda component 2']
               , edgecolors = color
               , s = 40
               , marker = marker
               , facecolors='none')

targets = ['Orange','Beige','White']
ax.legend(targets, prop={'size': 12})
ax.tick_params(labelsize=12)
ax.grid()
