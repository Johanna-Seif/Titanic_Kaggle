from process_data import process_data
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Learning Curves of the Models
from sklearn.model_selection import learning_curve

# TODO:
#   - clean data exploration

plt.close("all")

##########################################
#           Data Exploration             #
##########################################

def simple_visualisation(dataset):

    print(dataset.info())

    # Data description
    print("\nDescription of the numerical features:")
    print(dataset.describe())

    # Filling age
    dataset["Age"].fillna(value=dataset['Age'].median(), inplace=True)
    print("\nDescription of the numerical features after filling the age:")
    print(dataset.describe())

    # Set seaborn style
    sns.set(style='ticks')
    fg = sns.FacetGrid(data=dataset[dataset['Fare'] < 300], hue='Survived', col='Pclass', row='Parch', aspect=1)
    fg.map(plt.scatter, 'Age', 'Fare').add_legend()
    plt.show()

##############################################
#              Learning Curves               #
##############################################

# Learning curves adapted from
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimators, X, y, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimators : dictionary of estimators

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    number_models = len(estimators)
    fig, axes = plt.subplots(nrows=number_models, ncols=3)
    fig.tight_layout()
    axes = axes.reshape((number_models, 3))

    # Ploting figures for each estimator
    for row, estimator_key in enumerate(estimators):
        estimator = estimators[estimator_key]

        # Computing data for the curves
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[row, 0].grid()
        axes[row, 0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[row, 0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[row, 0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[row, 0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[row, 0].legend(loc="best")
        axes[row, 0].set_xlabel("Training examples")
        axes[row, 0].set_ylabel("Score")
        axes[row, 0].set_title("Learning Curves (" + estimator_key + ")")

        # Plot n_samples vs fit_times
        axes[row, 1].grid()
        axes[row, 1].plot(train_sizes, fit_times_mean, 'o-')
        axes[row, 1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[row, 1].set_xlabel("Training examples")
        axes[row, 1].set_ylabel("fit_times")
        axes[row, 1].set_title("Scalability of the model (" + estimator_key + ")")

        # Plot fit_time vs score
        axes[row, 2].grid()
        axes[row, 2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[row, 2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[row, 2].set_xlabel("fit_times")
        axes[row, 2].set_ylabel("Score")
        axes[row, 2].set_title("Performance of the model (" + estimator_key + ")")

    return plt
