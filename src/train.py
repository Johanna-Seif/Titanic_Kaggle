from process_data import load_and_process_data, prediction_to_csv
import numpy as np
import pandas as pd

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

constant_classifier = Constant_Classifier()
# Compute most common value
constant_classifier.find_most_common_output(dataset, 'Survived')
# Loading data test
data_test = pd.read_csv('../data/test.csv', header=0)
output = constant_classifier.classify(data_test, 'PassengerId')
prediction_to_csv(output, '../data/constant_classifier.csv')
