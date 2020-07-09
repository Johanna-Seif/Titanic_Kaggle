from process_data import process_data, prediction_to_csv
from visualisation import plot_learning_curve
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

# TODO:
#   - NN: tanh to have quicker descend
#   - Stochastic descend
#   - Verify correlation and importance of features

#####################################
#              Models               #
#####################################

models = {
    'LogisticRegression': LogisticRegression(),
    'rbf': svm.SVC(),
    'RandomForestClassifier': RandomForestClassifier(),
    'MLPClassifier': MLPClassifier()
}

params = {
    'LogisticRegression': {
        'C' : [.3, 0.6, 1, 1.5, 2, 4]
        },
    'rbf': {
        'kernel' : ['rbf']
        },
    'RandomForestClassifier': {
        'n_estimators' : [50, 100, 200],
        'max_depth' : [4, 5, 6, 7, 8, 9, 10, 15],
        'min_samples_leaf' : [1, 2, 3, 4, 6, 8, 10, 12]
        },
    'MLPClassifier': {
        'solver' : ['lbfgs', 'sgd'],
        'alpha' : [1e-7],
        'learning_rate' : ['constant', 'invscaling', 'adaptive'],
        'hidden_layer_sizes' : [(500,), (1000,)],
        'max_iter' : [400]
        }
}

##############################################
#              Training data                 #
##############################################

## Loading and processing data

# Merge to process
data_train = pd.read_csv('../data/train.csv', header=0)
data_test = pd.read_csv('../data/test.csv', header=0)
data_merge = pd.concat([data_train, data_test], keys=['train', 'test'])
dataset = process_data(data_merge)

# Extracting train test
X = dataset.loc['train'].drop(columns=['PassengerId', 'Survived'])
y = dataset.loc['train'].Survived

# Extracting data test
id = dataset.loc['test'].PassengerId
X_test = dataset.loc['test'].drop(columns=['PassengerId', 'Survived'])

# # Dividing in training and cross validation set
# X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)

# Table of estimators
helper = EstimatorSelectionHelper(models, params)
helper.fit(X, y)
score_summary = helper.score_summary()
score_summary.to_csv('../outputs/score_summary.csv')

# Creating dictionary of the best estimators
estimators = {}
for key in models:
    estimators[key] = helper.grid_searches[key].best_estimator_

# Predict for rbf
for model in models:
    prediction = estimators[model].predict(X_test)
    prediction_to_csv(id, prediction, f'../outputs/{model}_submission.csv')

##############################################
#              Learning Curves               #
##############################################

plt = plot_learning_curve(estimators,
                          X,
                          y,
                          train_sizes=np.linspace(.1, 1.0, 10))
plt.savefig('../outputs/learning_curves.pdf', bbox_inches='tight')
plt.show()
