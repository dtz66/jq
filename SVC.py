import pandas as pd
import jieba
import jieba.analyse
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

train_file = 'D:/桌面/BDCI-text-classification-master7/data/segments/web_train.csv'
test_file = 'D:/桌面/BDCI-text-classification-master7/data/segments/web_test.csv'
sample = pd.read_csv('D:/桌面/BDCI-text-classification-master7/data/sample.csv')

def read_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data


def tfidf_tags(train_data):
    tfidf = []
    for index, row in train_data.iterrows():
        label = row[0]
        comment = row[1]
        tags = jieba.analyse.extract_tags(comment, topK=10, withWeight=True)
        tags.insert(0, label)

        tfidf.append(tags)

    return tfidf


def aggre_tfidf(tfidf):
    freq_bad = {}
    for row in tfidf:
        label = row[0]
        freqs = row[1]
        if label == 1:
            for freq in freqs:
                word = freq[0]
                weight = freq[1]
                try:
                    if freq_bad[word] < weight:
                        freq_bad[word] = weight
                except KeyError:
                    freq_bad[word] = weight

    return freq_bad


train_data, test_data = read_data(train_file, test_file)
tfidf = tfidf_tags(train_data)

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(train_data['segments'])

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', SVC(C=1, kernel='linear'))])
text_clf = text_clf.fit(train_data['segments'], train_data['label'])
predicted = text_clf.predict(test_data['segments'])


sample.drop('label', axis = 1, inplace = True)
sample['label'] = predicted
sample.to_csv('D:/桌面/BDCI-text-classification-master7/data/predicts/predict4.csv', index=False)
