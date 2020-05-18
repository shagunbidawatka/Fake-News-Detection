# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:05:45 2020

@author: LENOVO
Fake News Detection Project
"""

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import random
import matplotlib.pyplot as plt
#Read the data
df=pd.read_csv('news.csv')
#Get shape and head
print(df.shape)
df.head()
labels=df.label
labels.head()

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier()
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score1=accuracy_score(y_test,y_pred)
print(f'Accuracy of Td-Tfd classifier: {round(score1*100,2)}%')
#DataFlair - Build confusion matrix
print("confusion matrix of Tf-Tdf classifier\n",confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))
cm1=confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
p1=[]
r1=[]
for label in range(2):
 #   print(f"{label:5d} {precision(label, cm1):9.2f} {recall(label, cm1):6.2f}")
    p1.append(round(precision(label, cm1),2))
    r1.append(round(recall(label, cm1),2))
print(classification_report(y_test,y_pred))


count_vectorizer=CountVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
count_train=count_vectorizer.fit_transform(x_train) 
count_test=count_vectorizer.transform(x_test)
#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier()
pac.fit(count_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(count_test)
score2=accuracy_score(y_test,y_pred)
print(f'\nAccuracy of count vectorizer: {round(score2*100,2)}%')
#DataFlair - Build confusion matrix
print("confusion matrix of count vectorizer classifier\n",confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))
cm2=confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
p2=[]
r2=[]
for label in range(2):
 #   print(f"{label:5d} {precision(label, cm2):9.2f} {recall(label, cm2):6.2f}")
    p2.append(round(precision(label, cm2),2))
    r2.append(round(recall(label, cm2),2))
print(classification_report(y_test,y_pred))

#Graph


index = np.arange(5)
bar_width = 0.35
tick_label=["Accuracy", "precision(y)", "precision(no)", "recall(yes)", "recall(no)"]
plt.ylim(0.8,1)
summer = plt.bar(index, [score1,p1[0],p1[1],r1[0],r1[1]], bar_width,
                label="Tf-Tdf",tick_label = tick_label)

winter = plt.bar(index+bar_width, [score2,p2[0],p2[1],r1[0],r1[1]],
                 bar_width, label="count")
plt.xlabel('Measure')
plt.ylabel('No.')
plt.title('analysis')
plt.xticks(index + bar_width / 2)

plt.legend()

plt.show()