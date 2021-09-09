from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import classification_report


def classify_sentiment(classifier,chi_train_corpus_tf_idf,chi_test_corpus_tf_idf,label_train,label_test):
    clf   = classifier
    clf.fit(chi_train_corpus_tf_idf,label_train)
    pred = clf.predict(chi_test_corpus_tf_idf)
    accuracy = clf.score(chi_test_corpus_tf_idf,label_test)
    cm = confusion_matrix(pred,label_test)
    f1 = f1_score(pred,label_test)
    return accuracy,f1,cm, pred, label_test




def classifier_train(chi_train_corpus_tf_idf,label_train,chi_test_corpus_tf_idf,label_test,train_choice,test_choice):

    rbf_parameters = [[0.9],[0.9],[0.9],[0.9],[0.9],[0.8],[0.9],[0.9],[0.8],[0.9],[0.9],[0.9]]

    val = (train_choice)*3 + test_choice

    Gamma = rbf_parameters[val][0]

    classifiers = [MultinomialNB()]

    
    accu = []
    
    classify = ["Multinomial NB"]

    for i in range(len(classifiers)):
        acc,f1,cm, pred, label_test = classify_sentiment(classifiers[i],chi_train_corpus_tf_idf,chi_test_corpus_tf_idf,label_train,label_test)

        accu.append(acc)

        print(classify[i]+" "+"F1 score adalah :",f1)
        print(classify[i]+" "+"Classification Report:")
        print(classification_report(label_test, pred))
        print("\n")

from sklearn.feature_extraction.text import TfidfVectorizer

vector_parameters = [[2,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],
[3,0.8],[3,0.8],[3,0.8]]


def featureextraction(train_corpus,test_corpus,label_train,train_choice,test_choice):

	val = (train_choice)*3 + test_choice

	param  = vector_parameters[val]
	mindf = param[0]
	maxdf = param[1]

	vectorizer = TfidfVectorizer(min_df=mindf,max_df=maxdf,use_idf=True,sublinear_tf=True,stop_words='english')

	train_corpus_tf_idf = vectorizer.fit_transform(train_corpus,label_train)

	test_corpus_tf_idf = vectorizer.transform(test_corpus)

	return [train_corpus_tf_idf,test_corpus_tf_idf]

from sklearn.feature_selection import SelectKBest,chi2



chi_square_parameters = [[5000],[4000],[4000],[4000],[4000],[4000],[4000],['all'],['all'],['all'],
['all'],['all'],[2500],['all']]


def featureselection(train_corpus_tf_idf,test_corpus_tf_idf,label_train,train_choice,test_choice):

	val = (train_choice)*3 + test_choice

	k = chi_square_parameters[val][0]

	if(k=='all'):
		K = train_corpus_tf_idf.shape[1]
	else:
		K = k 

	vectorizer_chi2 = SelectKBest(chi2,k=K)

	chi_train_corpus_tf_idf = vectorizer_chi2.fit_transform(train_corpus_tf_idf,label_train)

	chi_test_corpus_tf_idf = vectorizer_chi2.transform(test_corpus_tf_idf)

	return [chi_train_corpus_tf_idf,chi_test_corpus_tf_idf]

import re
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


train_list = ["Dataset/Actualdata/Books/Bookstrain.txt","Dataset/Actualdata/Dvd/Dvdtrain.txt","Dataset/Actualdata/Electronics/Electronicstrain.txt","Dataset/Actualdata/Kitchen/Kitchentrain.txt"]

test_list = ["Dataset/Actualdata/Books/Bookstest.txt","Dataset/Actualdata/Dvd/Dvdtest.txt","Dataset/Actualdata/Electronics/Electronicstest.txt","Dataset/Actualdata/Kitchen/Kitchentest.txt"]


stopword = stopwords.words('english') 

def preprocess(sentence):
	sentence = re.sub('[^\w\s]'," ",str(sentence))
	sentence = re.sub('[^a-zA-Z]'," ",str(sentence))
	sents = word_tokenize(sentence)
	new_sents = " "
	for i in range(len(sents)):
		if(sents[i].lower() not in stopword):
			new_sents+=sents[i].lower()+" "
	return new_sents

def preprocess_test(choice):

	the_file = test_list[choice]
	#print(the_file)
	with open(the_file,'r',encoding='utf-8') as f:
		test_data = f.readlines()

	corpus_test = []

	for i in range(400):
		sent = test_data[i]
		sent = sent[0:len(sent)-1]
		corpus_test.append(preprocess(sent))

	#print(corpus_test[0])

	label_test = np.zeros(400)
	label_test[0:200] = 1


	return [corpus_test,label_test]


def preprocess_train(choice):


	the_file = train_list[choice]
	#print(the_file)
	with open(the_file,'r',encoding='utf-8') as f:
		train_data = f.readlines()


	corpus_train = []

	for i in range(1600):
		sent = train_data[i]
		sent = sent[0:len(sent)-1]
		corpus_train.append(preprocess(sent))

	#print(corpus_train[0])

	label_train = np.zeros(1600)
	label_train[0:800] = 1

	return [corpus_train,label_train]

def preprocessing(train_choice,test_choice):

	corpus_train,label_train = preprocess_train(train_choice)

	corpus_test,label_test = preprocess_test(test_choice)

	return corpus_train,label_train,corpus_test,label_test

#ada 4 data yang bisa di test yaitu books,dvd,electronics,kitchen
#0 untuk books,1 untuk dvd,2 untuk electronics,3 untuk kitchen

train = 1

test = 2

	
train_choice,test_choice = train,test

pre_choice_train = train
pre_choice_test = test

corpus_train,label_train,corpus_test,label_test = preprocessing(pre_choice_train,pre_choice_test)

	
train_corpus_tf_idf,test_corpus_tf_idf = featureextraction(corpus_train,corpus_test,label_train,train_choice,test_choice)

chi_train_corpus_tf_idf,chi_test_corpus_tf_idf = featureselection(train_corpus_tf_idf,test_corpus_tf_idf,label_train,train_choice,test_choice)

classifier_train(chi_train_corpus_tf_idf,label_train,chi_test_corpus_tf_idf,label_test,train_choice,test_choice)
