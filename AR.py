import pandas as pd
from clean_text import CleanData
from apyori import apriori

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def extract_hashtags(text):
    tmp = {tag for tag in text.split() if tag.startswith("#")}
    return list(tmp)


def main():
    train = pd.read_csv('data/train.csv')
    # test = pd.read_csv('data/test.csv')

    train['hashtags'] = train['text'].apply(lambda x: extract_hashtags(x))

    train = train.drop(['keyword', 'location', 'id'], axis=1)

    cd = CleanData()

    data_clean = cd.normalize_text(train.copy())
    data_clean['transactions'] = data_clean['text'].str.split()
    data_clean['target'] = data_clean['target'].astype('str')

    data_clean['t'] = data_clean['transactions'] + data_clean['target'].apply(lambda x: [x])

    te = TransactionEncoder()
    te_ary = te.fit(data_clean['t']).transform(data_clean['t'])
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    # rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))

    one_rules = rules[(rules['consequents'] == {'1'})]

    one_rules = one_rules[(one_rules['confidence'] >= 0.7)]
    # one_rules = one_rules[(one_rules['antecedent_len'] >= 2) &
    #                   (one_rules['confidence'] > 0.75) &
    #                   (one_rules['lift'] > 1.2)]

    zero_rules = rules[(rules['consequents'] == {'0'})]
    zero_rules = zero_rules[(zero_rules['confidence'] >= 0.7)]
    print('done')


if __name__ == '__main__':
    main()

# records = []
# for i in len(data_clean):
#     records.append(data_clean['transactions'][i].append(data_clean['target'][0]))
#
# association_rules = apriori(data_clean['transactions'], min_support=0.1, min_confidence=0.1, min_lift=1, min_length=3)
# association_results = list(association_rules)
#
# print(len(association_results))
#
# print(association_results[0])
