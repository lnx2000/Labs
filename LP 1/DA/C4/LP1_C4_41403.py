# Twitter Data Sentiment Analysis

import pandas as pd
import numpy as np
import re 
import nltk 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Importing Dataset

train_data = pd.read_csv('./train_tweets.csv',encoding='latin1')
train_data

test_data = pd.read_csv('./test_tweets.csv',encoding='latin1')
test_data

print('Training Set Shape = {}'.format(train_data.shape))
print('Test Set Shape = {}'.format(test_data.shape))


# Converting Categorical Labels to Numeric Labels

# Removing URLs and HTML from Tweets

def remove_urls(text):
    url_remove = re.compile(r'https?://\S+|www\.\S+')
    return url_remove.sub(r'', text)
train_data['tweet']=train_data['tweet'].apply(lambda x:remove_urls(x))
test_data['tweet']=test_data['tweet'].apply(lambda x:remove_urls(x))

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
train_data['tweet']=train_data['tweet'].apply(lambda x:remove_html(x))
test_data['tweet']=test_data['tweet'].apply(lambda x:remove_html(x))

# Converting the Tweet text to lowercase

def lower(text):
    low_text= text.lower()
    return low_text
train_data['tweet']=train_data['tweet'].apply(lambda x:lower(x))
test_data['tweet']=test_data['tweet'].apply(lambda x:lower(x))

# Removing numerical values from Tweet text

def remove_num(text):
    remove= re.sub(r'\d+', '', text)
    return remove
train_data['tweet']=train_data['tweet'].apply(lambda x:remove_num(x))
test_data['tweet']=test_data['tweet'].apply(lambda x:remove_num(x))

# Removing Punctuation and Stopwords


from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))

def punct_remove(text):
    punct = re.sub(r"[^\w\s\d]","", text)
    return punct
train_data['tweet']=train_data['tweet'].apply(lambda x:punct_remove(x))
test_data['tweet']=test_data['tweet'].apply(lambda x:punct_remove(x))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
train_data['tweet']=train_data['tweet'].apply(lambda x:remove_stopwords(x))
test_data['tweet']=test_data['tweet'].apply(lambda x:remove_stopwords(x))


# Removing @ Mentions, # Hashtags, and Spaces

def remove_mention(x):
    text=re.sub(r'@\w+','',x)
    return text
train_data['tweet']=train_data['tweet'].apply(lambda x:remove_mention(x))
test_data['tweet']=test_data['tweet'].apply(lambda x:remove_mention(x))

def remove_hash(x):
    text=re.sub(r'#\w+','',x)
    return text
train_data['tweet']=train_data['tweet'].apply(lambda x:remove_hash(x))
test_data['tweet']=test_data['tweet'].apply(lambda x:remove_hash(x))

def remove_space(text):
    space_remove = re.sub(r"\s+"," ",text).strip()
    return space_remove
train_data['tweet']=train_data['tweet'].apply(lambda x:remove_space(x))
test_data['tweet']=test_data['tweet'].apply(lambda x:remove_space(x))

# Preprocessed Data

train_data

X_train, X_test, y_train, y_test = train_test_split(train_data.tweet, train_data.label, test_size=0.33, random_state=42)

# TF-IDF

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, stop_words='english')

train_tfidf = tfidf.fit_transform(X_train)
test_tfidf = tfidf.transform(X_test)

# Classifier #1: Multinomial Naive Bayes

nb = MultinomialNB()
nb.fit(train_tfidf, y_train)
nb_model = nb.predict(test_tfidf)

accuracy_score(y_test, nb_model)

nb_conf = confusion_matrix(y_test, nb_model)
ylabel = ["Actual Hate","Actual Not Hate"]
xlabel = ["Predicted Hate","Predicted Not Hate"] 
plt.figure(figsize=(15,6))
sns.heatmap(nb_conf, annot=True, xticklabels = xlabel, yticklabels = ylabel, linewidths=1, fmt='g')


# Classifier #2: Linear Support Vector

lsvc = LinearSVC()
lsvc.fit(train_tfidf, y_train)
lsvc_model = lsvc.predict(test_tfidf)

accuracy_score(y_test, lsvc_model)

lsvc_conf = confusion_matrix(y_test, nb_model)
ylabel = ["Actual Hate","Actual Not Hate"]
xlabel = ["Predicted Hate","Predicted Not Hate"] 
plt.figure(figsize=(15,6))
sns.heatmap(lsvc_conf, annot=True, xticklabels = xlabel, yticklabels = ylabel, linewidths=1, fmt='g')


# Classifier #3: Random Forest

rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(train_tfidf, y_train)
rfc_model = rfc.predict(test_tfidf)

accuracy_score(y_test, rfc_model)

rfc_conf = confusion_matrix(y_test, rfc_model)
ylabel = ["Actual Hate","Actual Not Hate"]
xlabel = ["Predicted Hate","Predicted Not Hate"] 
plt.figure(figsize=(15,6))
sns.heatmap(rfc_conf, annot=True, xticklabels = xlabel, yticklabels = ylabel, linewidths=1, fmt='g')
