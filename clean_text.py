import re

import nltk.corpus

nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')


class CleanData:
    def __init__(self):
        self.stop = stop

    def clean_text(self, df, text_field):
        df[text_field] = df[text_field].str.lower()
        df[text_field] = df[text_field].apply(
            lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
        return df

    def normalize_text(self, train):
        data_clean = self.clean_text(train, "text")
        data_clean['text'] = data_clean['text'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

        return data_clean
