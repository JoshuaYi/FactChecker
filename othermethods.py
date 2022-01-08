#Team 20
#Authors: Gerik Swenson, Joshua Yi
#CSE 472 Fall 2021
#Project 2: Type 1 - Fake News Detection

from operator import index
import numpy as np
import pandas as pd
import nltk; from nltk.corpus import stopwords
import sklearn; from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split; from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import string
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

nltk.download('stopwords')
df = pd.read_csv('train.csv')
#checking the descriptive statistics of the data types
#print(df.describe(include=['O']))

#print(df.groupby('Country (mentioned)').Label.value_counts())
#print(df[['Country (mentioned)', 'Label']].groupby(['Country (mentioned)'], as_index=False).mean())


df = df[df['Source'] != 'No data'] #clean the data
df['Country (mentioned)'] = pd.factorize(df['Country (mentioned)'])[0]
df['Fact-checked Article'] = df['Fact-checked Article'].str.extract('www.(\w+)\.')
df['Fact-checked Article'] = pd.factorize(df['Fact-checked Article'])[0]
df['Source'] = pd.factorize(df['Source'])[0]

def processing(text): #remove punctuation and detect english words
    cleaning = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    cleaning = ' '.join(cleaning)
    remPunc = [char for char in cleaning if char not in string.punctuation]
    remPunc = ''.join(remPunc)
    return remPunc.lower()

cv = CountVectorizer()
X = cv.fit_transform(df['Claim'].apply(processing))
tempdf = pd.DataFrame(X.todense(), columns=cv.get_feature_names())
pd.concat([df, tempdf], axis = 1)

x_train = df.drop(['Label', 'Review Date', 'Claim'], axis = 1)
y_train = df['Label']

#x_train, x_test, y_train, y_test = train_test_split(df1, df['Label'], test_size= 0.20, stratify = df['Label'], random_state = 0)
classifier = clf = RandomForestClassifier(n_estimators=100)
#print(x_train.shape)
classifier.fit(x_train, y_train)
pred = classifier.predict(x_train)
print(classification_report(y_train, pred))


test = pd.read_csv('test.csv')
test = test[test['Source'] != 'No data'] #clean the data
test['Country (mentioned)'] = pd.factorize(test['Country (mentioned)'])[0]
test['Fact-checked Article'] = test['Fact-checked Article'].str.extract('www.(\w+)\.')
test['Fact-checked Article'] = pd.factorize(test['Fact-checked Article'])[0]
test['Source'] = pd.factorize(test['Source'])[0]

cv = CountVectorizer()
X = cv.fit_transform(test['Claim'].apply(processing))
tempdf = pd.DataFrame(X.todense(), columns=cv.get_feature_names())
pd.concat([test, tempdf], axis = 1)

x_test = test.drop(['Review Date', 'Claim'], axis = 1)
pred = classifier.predict(x_test)
idxList = list(range(1, 712))
prediction = pd.DataFrame(list(zip(*[idxList, pred])), columns=(['id','Predicted'])).to_csv('submission.csv', index=False)