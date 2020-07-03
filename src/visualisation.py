from process_data import load_and_process_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.close("all")

# Loading preprocessed data
dataset = load_and_process_data('../data/train.csv')

##############################################
#           Data visualisation               #
##############################################

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
