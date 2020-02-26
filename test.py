import pandas as pd
import spacy
from spacy import displacy
from collections import Counter
from tqdm import tqdm
from spacy.tokens import DocBin
from clean_text import CleanData

nlp = spacy.load('en_core_web_sm')


def doc_entity(docs, keywords):
    # docs = docs[docs['keywords'].map(lambda d: len(d)) > 0]

    results = pd.DataFrame()
    for index, doc in tqdm(docs.iterrows(), total=docs.shape[0]):
        row_dic = {}
        for keyword in doc['keywords']:
            if keyword in keywords:
                entity = keywords[keyword]
                if entity in row_dic:
                    row_dic.update({entity: row_dic[entity] + 1})
                else:
                    row_dic.update({entity: 1})
        if len(row_dic) < 1:
            continue
        row_dic.update({'id': doc['id']})
        results = results.append(row_dic, ignore_index=True)

    results.fillna(0, inplace=True)
    return results


def main():
    keywords_df = pd.read_csv('data/keywords.csv')
    keywords_dic = dict(zip(keywords_df['keyword'], keywords_df['entity']))
    data = pd.read_csv('data/taged_all.csv')

    cd = CleanData()
    data_clean = cd.normalize_text(data.copy())
    data_clean['keywords'] = data_clean['clean_text'].str.split()

    doc_entity_df = doc_entity(data_clean, keywords_dic)

    doc_entity_df = doc_entity_df.merge(data_clean[['id', 'target', 'predict']], how='left', left_on='id',
                                        right_on='id')
    doc_entity_df.set_index('id', inplace=True)
    doc_entity_df.to_csv('data/doc_entity_df.csv', index=True, header=True)

    doc_bin = DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"], store_user_data=True)
    texts = [
        "Disaster control teams are studying ways to evacuate the port area in response to tidal wave warnings.[900037]"]
    nlp = spacy.load("en_core_web_md")
    for doc in nlp.pipe(texts):
        doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()

    # Read and write binary file
    with open('data/sample', "wb") as out_file:
        out_file.write(bytes_data)

    with open('data/sample', "rb") as in_file:
        data = in_file.read()
        in_file.close()

    # Deserialize later, e.g. in a new process
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_bytes(data)
    docs = list(doc_bin.get_docs(nlp.vocab))

    # ###################################################################################
    data = pd.read_csv('data/taged_all.csv')

    for row in tqdm(data['text'], total=data.shape[0]):
        doc = nlp(row)
        doc.to_disk('data/sample')
    print([(X.text, X.label_) for X in doc.ents])


if __name__ == '__main__':
    main()
