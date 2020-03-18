import pandas as pd
from clean_text import CleanData
# from apyori import apriori

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import spacy

from tqdm import tqdm


def extract_hashtags(text):
    tmp = {tag for tag in text.split() if tag.startswith("#")}
    return list(tmp)


def text_to_vector_ar(data_clean, ones_keywords, ones_hashtags, zeros_keywords, zeros_hashtags):
    results = pd.DataFrame()
    for index, row in tqdm(data_clean.iterrows(), total=data_clean.shape[0]):
        row_dic = {}
        for f in ones_keywords:
            if f in row['keywords']:
                row_dic.update({f: 1})
            else:
                row_dic.update({f: 0})
        for f in ones_hashtags:
            if f in row['hashtags']:
                row_dic.update({f: 4})
            else:
                row_dic.update({f: 0})

        for f in zeros_keywords:
            if f in row['keywords']:
                row_dic.update({f: -1})
            else:
                row_dic.update({f: 0})
        for f in zeros_hashtags:
            if f in row['hashtags']:
                row_dic.update({f: -4})
            else:
                row_dic.update({f: 0})

        results = results.append(row_dic, ignore_index=True)

    return results


def text_to_vector_ar_all(data_clean, rule_keywords, keywords):
    results = pd.DataFrame()
    for index, row in tqdm(data_clean.iterrows(), total=data_clean.shape[0]):
        row_dic = {}
        for f in row['keywords']:
            if f in rule_keywords:
                row_dic.update({f: 1})
            else:
                row_dic.update({f: 0})
        for f in ones_hashtags:
            if f in row['hashtags']:
                row_dic.update({f: 4})
            else:
                row_dic.update({f: 0})

        for f in zeros_keywords:
            if f in row['keywords']:
                row_dic.update({f: -1})
            else:
                row_dic.update({f: 0})
        for f in zeros_hashtags:
            if f in row['hashtags']:
                row_dic.update({f: -4})
            else:
                row_dic.update({f: 0})

        results = results.append(row_dic, ignore_index=True)

    return results


def text_to_vector_weighted_entity(data_clean, keywords):
    results = pd.DataFrame(columns=keywords.keys())
    for index, row in tqdm(data_clean.iterrows(), total=data_clean.shape[0]):
        row_dic = {}
        for f in row['keywords']:
            if f in keywords:
                row_dic.update({f: keywords[f]})
        if len(row_dic) > 0:
            row_dic.update({'id': row['id']})
            results = results.append(row_dic, ignore_index=True)

    results.fillna(0, inplace=True)
    return results


def extract_ents(keywords):
    nlp = spacy.load('en_core_web_md')
    key_ents = {}
    for key in tqdm(keywords):
        doc = nlp(key)
        if len(doc.ents) > 0:
            key_ents.update({key: doc.ents[0].label_})

    return key_ents


def main():
    train = pd.read_csv('data/train.csv')
    # test = pd.read_csv('data/test.csv')
    entity_weight = pd.read_csv('data/entity_weight.csv')

    train = train.drop(['keyword', 'location'], axis=1)

    # train_hashtags = train.copy()
    train['hashtags'] = train['text'].apply(lambda x: extract_hashtags(x))
    train_hashtags = train[train['hashtags'].map(lambda d: len(d)) > 0].copy()
    train_hashtags['target'] = train_hashtags['target'].astype(str)
    train_hashtags['t'] = train_hashtags['hashtags'] + train_hashtags['target'].apply(lambda x: [x])

    # hashtags = []
    # for x in train['hashtags']:
    #     hashtags.extend(x)
    #
    # hashtags = list(set(hashtags))

    cd = CleanData()
    data_clean = cd.normalize_text(train.copy())
    data_clean['keywords'] = data_clean['clean_text'].str.split()
    data_clean['target'] = data_clean['target'].astype('str')

    # keywords of all rows
    keys = []
    data_clean['keywords'].apply(lambda x: keys.extend(x))
    keys = list(set(keys))
    # keywords = extract_ents(keys)
    # (pd.DataFrame.from_dict(keywords, orient='index')).to_csv('data/keywords.csv')
    keywords = pd.read_csv('data/keywords.csv')

    keywords = keywords.merge(entity_weight, how='left', left_on='entity', right_on='entity')
    keywords_dic = dict(zip(keywords['keyword'], keywords['weight']))

    # messages_vector = text_to_vector_weighted_entity(data_clean, keywords_dic)
    # messages_vector.set_index('id').to_csv('data/messages_vector.csv', header=True)
    data_clean['t'] = data_clean['keywords'] + data_clean['target'].apply(lambda x: [x])

    te = TransactionEncoder()
    te_ary = te.fit(data_clean['t']).transform(data_clean['t'])
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)

    rules = association_rules(frequent_itemsets)
    # rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules = rules[((rules['consequents'] == {'1'}) | (rules['consequents'] == {'0'})) & (rules['confidence'] >= 0.55)]
    tmp = rules[(rules['consequents'] == {'1'}) | (rules['consequents'] == {'0'})]
    rule_keywords = list(rules['antecedents'])
    rule_keywords = frozenset().union(*rule_keywords)

    with open('data/rule_keywords.txt', 'w') as f:
        for item in rule_keywords:
            f.write("%s\n" % item)

    rule_entity_keywords = {key: value for key, value in keywords_dic.items() if key in rule_keywords}

    # one_rules = rules[((rules['consequents'] == {'1'}) | (rules['consequents'] == {'0'})) & (rules['confidence'] >= 0.55)]
    # one_rules = one_rules[(one_rules['confidence'] >= 0.55)]
    # one_rules = one_rules[(one_rules['antecedent_len'] >= 2) &
    #                   (one_rules['confidence'] > 0.75) &
    #                   (one_rules['lift'] > 1.2)]

    # zero_rules = rules[(rules['consequents'] == {'0'})]
    # zero_rules = zero_rules[(zero_rules['confidence'] >= 0.55)]
    #
    # # hashtag rules
    # te_hashtags = TransactionEncoder()
    # te_ary_hashtags = te_hashtags.fit(train_hashtags['t']).transform(train_hashtags['t'])
    # df_hashtags = pd.DataFrame(te_ary_hashtags, columns=te_hashtags.columns_)
    # frequent_itemsets_hashtags = apriori(df_hashtags, min_support=0.005, use_colnames=True)
    #
    # rules_hashtags = association_rules(frequent_itemsets_hashtags, metric="confidence", min_threshold=0.6)
    # # rules_hashtags = association_rules(frequent_itemsets_hashtags, metric="lift", min_threshold=1.2)
    #
    # rules_hashtags["antecedent_len"] = rules_hashtags["antecedents"].apply(lambda x: len(x))
    #
    # one_rules_hashtags = rules_hashtags[(rules_hashtags['consequents'] == {'1'})]
    # one_rules_hashtags = one_rules_hashtags[(one_rules_hashtags['confidence'] >= 0.55)]
    #
    # zero_rules_hashtags = rules_hashtags[(rules_hashtags['consequents'] == {'0'})]
    # zero_rules_hashtags = zero_rules_hashtags[(zero_rules_hashtags['confidence'] >= 0.55)]
    #
    # # frozensets of keywords and hashtags
    # ones_keywords = list(one_rules['antecedents'])
    # ones_keywords = frozenset().union(*ones_keywords)
    # ones_hashtags = list(one_rules_hashtags['antecedents'])
    # ones_hashtags = frozenset().union(*ones_hashtags)
    #
    # zeros_keywords = list(zero_rules['antecedents'])
    # zeros_keywords = frozenset().union(*zeros_keywords)
    # zeros_hashtags = list(zero_rules_hashtags['antecedents'])
    # zeros_hashtags = frozenset().union(*zeros_hashtags)

    # vector of messages
    messages_vector = text_to_vector_ar(data_clean, ones_keywords, ones_hashtags, zeros_keywords, zeros_hashtags)
    X_train_df, X_test_df, y_train, y_test = train_test_split(data_clean['text'], data_clean['target'], random_state=0)

    X_train = messages_vector.iloc[X_train_df.index].values
    X_test = messages_vector.iloc[X_test_df.index].values

    # train model
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print('LogisticRegression scores:\n')
    print(classification_report(y_test, y_predict))

    clf_svm = svm.SVC()
    clf_svm.fit(X_train, y_train)
    y_predict = clf_svm.predict(X_test)
    print('SVM Results:\n')
    print(classification_report(y_test, y_predict))

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
