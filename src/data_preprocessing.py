import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


def load_data(data_path):  #data path : 'data/raw/dataset.csv'
    df = pd.read_csv(data_path)

    train, valid, test = np.split(df.sample(frac=1), 
                              [int(0.6*len(df)), 
                               int(0.8*len(df))])
    return train, valid, test


def scale_dataset(dataframe, name:str, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    #scaling the values in the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    """
    where there are more rows for class == 0 than class == 1, we need a way to balance
    hence we use oversample which takes more of the less class
    to increase the size of the dataset of that smaller dataset"""

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)


    data = np.hstack((X, np.reshape(y, (-1,1))))

    try:
        
       # Define the file path, for example, in the user's home directory
       # # The `~` represents the home directory
       save_path = os.path.join('data/split_data')
       os.makedirs(save_path, exist_ok=True)


       file_path1 = os.path.join(save_path, f'{name}_data.npy')
       np.save(file_path1 , data)

       file_path2 = os.path.join(save_path, f'X_{name}.npy')
       np.save(file_path2, X)

       file_path3 = os.path.join(save_path, f'y_{name}.npy')
       np.save(file_path3, y)
        
    #error handline
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

    return data, X, y

def main():
    train, valid, test = load_data("data/raw/dataset.csv")

    train, X_train, y_train = scale_dataset(train, "train", oversample=True)
    valid, X_valid, y_valid = scale_dataset(valid, "valid", oversample=False)
    test, X_test, y_test = scale_dataset(test, "test", oversample=False)


if __name__ == "__main__":
    main()