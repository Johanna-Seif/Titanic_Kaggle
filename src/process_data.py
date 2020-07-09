import pandas as pd
import numpy as np
# Feature scaling
from sklearn.preprocessing import StandardScaler

# TODO
#   - Adding polynomial features

# Loading and processing training set
def process_data(data):
    ''' Apply preprocessing '''

    ## Processing data

    # Guessing missing values for Age
    data['Age'] = data.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
    # Guessing missing values for Fare
    data['Fare'] = data.groupby(['Pclass', 'Sex'])['Fare'].apply(lambda x: x.fillna(x.median()))
    # Guessing missing values for Embarked
    data['Embarked'].fillna(data.Embarked.mode()[0], inplace = True)

    # Taking of Cabin because too sparse
    del data['Cabin']
    # Taking off ticket, not sure of its use
    del data['Ticket']

    # Extracting title from the name
    data['Title'] = data['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
    titles_dict = {
        'Mr' : 'Mr',
        'Mrs' : 'Mrs',
        'Miss' : 'Miss',
        'Master' : 'Master',
        'Don' : 'Nobility',
        'Rev' : 'Profession',
        'Dr' : 'Profession',
        'Mme' : 'Mrs',
        'Ms' : 'Mrs',
        'Major' : 'Profession',
        'Lady' : 'Nobility',
        'Sir' : 'Nobility',
        'Mlle' : 'Miss',
        'Col' : 'Profession',
        'Capt' : 'Profession',
        'the Countess' : 'Nobility',
        'Jonkheer' : 'Nobility',
        'Dona' : 'Nobility'
    }
    data.Title = data.Title.apply(lambda x : titles_dict[x])
    # Drop Name
    data.drop(columns=['Name'], inplace=True)

    # Creating dummy features for Sex, Pclass and Embarked
    data = pd.get_dummies(data, columns=['Sex'])
    data = pd.get_dummies(data, columns=['Pclass'])
    data = pd.get_dummies(data, columns=['Embarked'])
    data = pd.get_dummies(data, columns=['Title'])

    # Feature scaling
    scaler = StandardScaler()
    data[['Age','Fare']] = scaler.fit_transform(data[['Age','Fare']])

    # Verify missing values
    missing_values = data.drop(columns=['Survived']).isna().sum().sum()
    print(f'Processed everything. Missing values left: {missing_values}')

    return data


def prediction_to_csv(id, prediction, file_path):
    ''' Creating submission file '''
    # output_array =
    prediction = prediction.reshape((prediction.shape[0], 1))
    id = id.to_numpy().reshape((prediction.shape[0], 1))
    output_array = np.concatenate((id, prediction), axis=1)
    np.savetxt(file_path, output_array, fmt='%i', delimiter=',', header='PassengerId,Survived', comments='')
