import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

from clean_text import CleanData


def train_and_test_model(X_train, y_train, X_test):
    pipeline_sgd = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('nb', SGDClassifier()),
    ])
    model = pipeline_sgd.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    return {'model': model, 'y_predict': y_predict}


def submission_test(test, model, cd):
    submission_test_clean = test.copy()
    submission_test_clean = cd.clean_text(submission_test_clean, "text")
    submission_test_clean['text'] = submission_test_clean['text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (cd.stop)]))
    submission_test_clean = submission_test_clean['text']

    submission_test_pred = model.predict(submission_test_clean)

    id_col = test['id']
    submission_df_1 = pd.DataFrame({
        "id": id_col,
        "target": submission_test_pred})

    return submission_df_1


def towardsdatascience(X_train, X_test, y_train, y_test, test, cd):
    # normalize text (lower down, delete url, delete # from hashtags)

    # train and test model
    results = train_and_test_model(X_train, y_train, X_test)
    model = results['model']
    y_predict = results['y_predict']
    print(classification_report(y_test, y_predict))

    # submission test data with model
    submission_df_1 = submission_test(test, model, cd)
    return submission_df_1


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    train = train.drop(['keyword', 'location'], axis=1)

    cd = CleanData()

    data_clean = cd.clean_text(train.copy(), 'text')
    # data_clean.to_csv('data/data_clean.csv', index=False, header=True)

    # select train and test data
    X_train, X_test, y_train, y_test = train_test_split(data_clean['text'], data_clean['target'], random_state=0)

    results = train_and_test_model(data_clean['text'], data_clean['target'], data_clean['text'])
    y_predict = results['y_predict']
    print(classification_report(data_clean['target'], y_predict))
    submission = pd.DataFrame(y_predict)
    submission['id'] = train['id']
    submission.to_csv('data/submission_all.csv', index=False, header=True)

    submission_df_1 = towardsdatascience(X_train, X_test, y_train, y_test, test, cd)
    submission_df_1.to_csv('data/submission_1.csv', index=False)

    print('done')
