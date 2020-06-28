import pandas as pd
import numpy as np

# Loading training set
def load_and_process_data(file_path):
    ''' Load the training data and apply preprocessing '''
    # Loading csv file
    data = pd.read_csv(file_path, header=0)
    print('Data loaded.')

    return data


def prediction_to_csv(output_array, file_path):
    np.savetxt(file_path, output_array, fmt='%i', delimiter=',', header='PassengerId,Survived', comments='')
