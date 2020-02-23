import pandas as pd
import re
import nltk.corpus

nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df


def normalize_text(train):
    data_clean = clean_text(train, "text")
    data_clean['text'] = data_clean['text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return data_clean


def train_and_test_model(X_train, y_train, X_test, y_test):
    pipeline_sgd = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('nb', SGDClassifier()),
    ])
    model = pipeline_sgd.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    return {'model': model, 'y_predict': y_predict}


def submission_test(test, model):
    submission_test_clean = test.copy()
    submission_test_clean = clean_text(submission_test_clean, "text")
    submission_test_clean['text'] = submission_test_clean['text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    submission_test_clean = submission_test_clean['text']

    submission_test_pred = model.predict(submission_test_clean)

    id_col = test['id']
    submission_df_1 = pd.DataFrame({
        "id": id_col,
        "target": submission_test_pred})

    return submission_df_1


def towardsdatascience(X_train, X_test, y_train, y_test, test):
    # normalize text (lower down, delete url, delete # from hashtags)

    # train and test model
    results = train_and_test_model(X_train, y_train, X_test, y_test)
    model = results['model']
    y_predict = results['y_predict']
    print(classification_report(y_test, y_predict))

    # submission test data with model
    submission_df_1 = submission_test(test, model)
    return submission_df_1


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    train = train.drop(['keyword', 'location', 'id'], axis=1)
    data_clean = normalize_text(train)
    # data_clean.to_csv('data/data_clean.csv', index=False, header=True)

    # select train and test data
    X_train, X_test, y_train, y_test = train_test_split(data_clean['text'], data_clean['target'], random_state=0)

    submission_df_1 = towardsdatascience(X_train, X_test, y_train, y_test, test)
    # submission_df_1.to_csv('data/submission_1.csv', index=False)

    print('done')
