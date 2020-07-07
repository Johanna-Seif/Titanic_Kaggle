from process_data import load_and_process_data, prediction_to_csv
import numpy as np
import pandas as pd

# Scikit Models
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Estimators
from utils import EstimatorSelectionHelper

#####################################
#              Models               #
#####################################

models = {
    'LogisticRegression': LogisticRegression(),
    'rbf': svm.SVC(),
    # 'RandomForestClassifier': RandomForestClassifier(),
    # 'MLPClassifier': MLPClassifier()
}

params = {
    'LogisticRegression': {
        'C' : [1]},
    'rbf': {
        'kernel' : ['rbf']},
    'RandomForestClassifier': {
        'max_depth' : [6, 7, 8, 9, 10, 15],
        'min_samples_leaf' : [1, 2, 3, 4, 6, 8, 10]},
    'MLPClassifier': {
        'solver' : ['lbfgs'], 'alpha' : [1e-7], 'hidden_layer_sizes' : [(1000,)]}
}

##############################################
#              Training data                 #
##############################################


# Loading preprocessed data
dataset = load_and_process_data('../data/train.csv')
#print(dataset.info())

# Defining features and labels
features_cols = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
X = dataset.loc[:, features_cols]
y = dataset.Survived

# # Dividing in training and cross validation set
# X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

# Table of estimators
helper = EstimatorSelectionHelper(models, params)
helper.fit(X, y)
helper.score_summary()

# Loading data test
X_test = load_and_process_data('../data/test.csv')
id = X_test.PassengerId

# # Predict for rbf
# prediction = helper.grid_searches['rbf'].predict(X_test.loc[:, features_cols])
# prediction_to_csv(id, prediction, '../data/give_a_name_submission.csv')

##############################################
#              Learning Curves               #
##############################################
