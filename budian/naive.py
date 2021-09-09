import pandas as pd
import numpy as np
import math as math

truenews = pd.read_csv('true-split.csv')
fakenews = pd.read_csv('fake-split.csv')

truenews['True/Fake']='True'
fakenews['True/Fake']='Fake'

# gabung fake & true jadi 1
news = pd.concat([truenews, fakenews])
news["Article"] = news["title"] + news["text"]
news.sample(frac = 1)


#Data Cleaning
from nltk.corpus import stopwords
import string

def process_text(s):

    # cek tanda baca
    nopunc = [char for char in s if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_string

news['Clean Text'] = news['Article'].apply(process_text)
news.sample(5)
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=process_text).fit(news['Clean Text'])

print(len(bow_transformer.vocabulary_))
news_bow = bow_transformer.transform(news['Clean Text'])

sparsity = (100.0 * news_bow.nnz / (news_bow.shape[0] * news_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(news_bow)
news_tfidf = tfidf_transformer.transform(news_bow)
print(news_tfidf.shape)


#Train Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
fakenews_detect_model = MultinomialNB().fit(news_tfidf, news['True/Fake'])

#Model Evaluation
predictions = fakenews_detect_model.predict(news_tfidf)
print(predictions)

from sklearn.metrics import classification_report
print (classification_report(news['True/Fake'], predictions))


from sklearn.model_selection import train_test_split

news_train, news_test, text_train, text_test = train_test_split(news['Article'], news['True/Fake'], test_size=0.3)

print(len(news_train), len(news_test), len(news_train) + len(news_test))


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)), 
    ('tfidf', TfidfTransformer()), 
    ('classifier', MultinomialNB()),
])
pipeline.fit(news_train,text_train)

# Hasil Akhir
predictions = pipeline.predict(news_test)
print(classification_report(predictions,text_test))