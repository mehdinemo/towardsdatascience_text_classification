import pandas as pd
from clean_text import CleanData
import numpy as np


def text_to_allkeywords(data: pd.DataFrame) -> pd.DataFrame:
    data['transactions'] = data['clean_text'].str.split()

    allkeywords = pd.DataFrame({'message_id': np.repeat(data['id'].values, data['transactions'].str.len()),
                                'word': np.concatenate(data['transactions'].values)})
    allkeywords['count'] = 1
    allkeywords = allkeywords.groupby(['message_id', 'word'], as_index=False).sum()

    return allkeywords


def build_allkeywords():
    train = pd.read_csv('data/train.csv')

    train = train.drop(['keyword', 'location'], axis=1)

    cd = CleanData()

    data_clean = cd.normalize_text(train)
    allkeywords = text_to_allkeywords(data_clean)

    tmp = allkeywords.merge(allkeywords, how='inner', left_on='word', right_on='word')
    tmp.to_csv('data/allkeywords.csv')
    # tmp = tmp.groupby(['message_id_x', 'message_id_y'], as_index=False)['word'].sum()

    print('done')

def main():


if __name__ == '__main__':
    main()
