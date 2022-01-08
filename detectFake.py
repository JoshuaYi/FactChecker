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
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB


nltk.download('stopwords')
df = pd.read_csv('train.csv')
df = df[df['Source'] != 'No data'] #clean the data of no source

#testing whether these have an affect on overall performance.
#df['Fact-checked Article'] = df['Fact-checked Article'].str.extract('www.(\w+)\.')
#df['Fact-checked Article'] = pd.factorize(df['Fact-checked Article'])[0]
#df['Fact-checked Article'] = df['Fact-checked Article'].apply(str)

df['Important'] = df['Source'] + ' ' + df['Claim']  #+ ' ' + df['Fact-checked Article'] + ' ' + df['Country (mentioned)']

def processing(text): #remove punctuation and detect english words
    cleaning = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    cleaning = ' '.join(cleaning)
    remPunc = [char for char in cleaning if char not in string.punctuation]
    remPunc = ''.join(remPunc)
    return remPunc.lower()

df['Important'] = df['Important'].apply(processing)
dfI = df['Important'].values
dfL = df['Label'].values
cv = CountVectorizer()
cv.fit(dfI)
dfI = cv.transform(dfI)

x_train, x_test, y_train, y_test = train_test_split(dfI, dfL, test_size= .19, stratify = dfL)
classifier = MultinomialNB()

classifier.fit(x_train, y_train)
pred = classifier.predict(x_train)
print(classification_report(y_train, pred))


#take in unseen data
testDF = pd.read_csv('test.csv')
testDF = testDF[testDF['Source'] != 'No data'] #clean the data
testDF['Important'] = testDF['Source'] + '- ' + testDF['Claim'] #determin important factors
testDF['Important'] = testDF['Important'].apply(processing) #take out stopwords and punctuation
dfT = testDF['Important'].values
dfT = cv.transform(dfT) #transform unseen data
pred = classifier.predict(dfT) #predict labels for unseen data
print(pred)
idxList = list(range(1, 712))
print(idxList)
prediction = pd.DataFrame(list(zip(*[idxList, pred])), columns=(['id','Predicted'])).to_csv('submission.csv', index=False)