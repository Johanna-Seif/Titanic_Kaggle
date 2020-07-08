import pandas as pd
import numpy as np

# TODO
#   - dummy variables for strings
#   - feature scaling
#   - Adding polynomial features

# Loading training set
def load_and_process_data(file_path):
    ''' Load the training data and apply preprocessing '''
    # Loading csv file
    data = pd.read_csv(file_path, header=0)
    print('Data loaded.')

    # Processing data
    # Taking off PassengerId, Name, Embarked, Ticket
    del data['Name']
    del data['Ticket']
    del data['Embarked']
    # Taking off SibSp to avoid redundancy
    del data['SibSp']
    # Taking of Cabin because too sparse
    del data['Cabin']

    # Filling age and fare with the median
    data['Age'].fillna(value=data['Age'].median(), inplace=True)
    data['Fare'].fillna(value=data['Fare'].median(), inplace=True)

    # Changing string to int in Sex
    mymap = {'female' : 1, 'male' : 2}
    data = data.applymap(lambda s: mymap.get(s) if s in mymap else s)

    print('Data processed.')

    return data


def prediction_to_csv(id, prediction, file_path):
    ''' Creating submission file '''
    # output_array =
    prediction = prediction.reshape((prediction.shape[0], 1))
    id = id.to_numpy().reshape((prediction.shape[0], 1))
    output_array = np.concatenate((id, prediction), axis=1)
    np.savetxt(file_path, output_array, fmt='%i', delimiter=',', header='PassengerId,Survived', comments='')
