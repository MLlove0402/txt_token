import numpy as np 
import re
import nltk 
nltk.download("stopwords")
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords

data = load_files("/content/drive/My Drive/txt_sentoken")
X,y = data.data,data.target

documents = []

from nltk.stem.porter import *
stemmer = PorterStemmer()

for i in range(len(X)):
  document = re.sub(r'\W', ' ', str(X[i]))
  document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
  document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
  document = re.sub(r'\s+', ' ', document, flags=re.I)
  document = re.sub(r'^b\s+', '', document)
  
  document = document.lower()
  
  document = document.split()
  
  document= [stemmer.stem(word) for word in document]
  
  document = ' '.join(document)
  documents.append(document)
  
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
  
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))