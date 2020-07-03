from process_data import load_and_process_data, prediction_to_csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class Constant_Classifier():
    """Training a basic classifier class that extract the most common class"""

    def __init__(self):
        self.name = 'Constant classifier'
        self.value = 0

    def find_most_common_output(self,data,column_name):
        # Select survived column
        output_column = data.loc[:, column_name]
        self.value = output_column.mode()

    def classify(self, data_frame, id_name):
        id_column = data_frame.loc[:, id_name]
        id_column = id_column.to_numpy().reshape((data_frame.shape[0], 1))
        # Array of output
        classified_data = np.full((data_frame.shape[0], 1), self.value)
        return np.concatenate((id_column, classified_data), axis=1)

# Loading preprocessed data
dataset = load_and_process_data('../data/train.csv')
print(dataset.info())

# Defining features and labels
features_cols = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
X = dataset.loc[:, features_cols]
y = dataset.Survived
# Dividing in training test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

##############################################
#              Training data                 #
##############################################

# Training data with constant classifier
#---------------------------------------

# constant_classifier = Constant_Classifier()
# # Compute most common value
# constant_classifier.find_most_common_output(dataset, 'Survived')
# # Loading data test
# data_test = pd.read_csv('../data/test.csv', header=0)
# output = constant_classifier.classify(data_test, 'PassengerId')
# prediction_to_csv(output, '../data/constant_classifier.csv')

# Training data with LogisticRegression
#---------------------------------------

clf_LR = LogisticRegression(random_state=0, C=1)
# clf_LR.fit(X_train, y_train)
# print(clf_LR.score(X_train, y_train))
# print(clf_LR.score(X_test, y_test))
clf_LR.fit(X, y)
# Loading data test
X_test = load_and_process_data('../data/test.csv')
prediction = clf_LR.predict(X_test.loc[:, features_cols])
id = X_test.PassengerId

prediction_to_csv(id, prediction, '../data/LR_classifier_submission.csv')

# Training data with svm
#------------------------------

# clf_SVM = svm.SVC(kernel='rbf')
# clf_SVM.fit(X_train, y_train)
# print(clf_SVM.score(X_train, y_train))
# print(clf_SVM.score(X_test, y_test))
