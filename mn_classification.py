import pandas as pd
from clean_text import CleanData

from sklearn.model_selection import train_test_split


def main():
    print('done')


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    train = train.drop(['keyword', 'location', 'id'], axis=1)

    cd = CleanData()

    data_clean = cd.clean_text(train, 'text')
    # data_clean.to_csv('data/data_clean.csv', index=False, header=True)

    # select train and test data
    X_train, X_test, y_train, y_test = train_test_split(data_clean['text'], data_clean['target'], random_state=0)

    main()
