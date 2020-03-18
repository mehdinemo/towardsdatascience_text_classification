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
    tmp = allkeywords.merge(train[['id', 'target']], how='left', left_on='message_id', right_on='id')
    tmp = tmp.groupby(['word', 'target'], as_index=False)['count'].sum()
    tmp_1 = tmp.groupby(['word'], as_index=False)['count'].sum()

    word_confidence = tmp_1.merge(tmp[tmp['target'] == 1], how='left', left_on='word', right_on='word')
    word_confidence.drop(['target'], axis=1, inplace=True)
    word_confidence = word_confidence[word_confidence['count_x'] > 1]
    word_confidence.sort_values(by=['count_x'], inplace=True)
    word_confidence.to_csv(r'data/word_confidence.csv', index=False)
    word_confidence.fillna(0, inplace=True)

    word_confidence['confidence'] = word_confidence['count_y'] / word_confidence['count_x']

    allkeywords = allkeywords.merge(word_confidence[['word', 'confidence']], how='right', left_on='word',
                                    right_on='word')

    graph = allkeywords.merge(allkeywords, how='inner', left_on='word', right_on='word')
    # tmp.to_csv('data/allkeywords.csv')
    graph = graph.groupby(['message_id_x', 'message_id_y'], as_index=False)['confidence_x'].mean()
    graph.columns = ['soyrce', 'target', 'weight']
    graph.to_csv(r'data/graph.scv', index=False)
    print('done')


def main():
    build_allkeywords()
    print('done')


if __name__ == '__main__':
    main()
