import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('2010-capitalbikeshare-tripdata.csv')
data
data.info() 

data.dtypes   ##### info about types

data.describe()  #### statistics

#Drop unnecessary columns
data=data.drop('Start date',axis=1)
data=data.drop('End date',axis=1)
data=data.drop('Start station',axis=1)
data=data.drop('End station',axis=1)

data.head() 

########### label encoder :- Converts Data(Strings) into numbers for using it for calculations 
le = LabelEncoder()
le.fit(data['Member type'])
data['Member type'] = le.transform(data['Member type'])

data.head()

le = LabelEncoder()
le.fit(data['Bike number'])
data['Bike number'] = le.transform(data['Bike number'])

data.head()

data.shape     ###### data.size

#train=np.array(data.iloc[0:85000])   ### spitting into training and tetsign
#test=np.array(data.iloc[85000:,])

train,test=train_test_split(data,test_size=0.20,random_state=0)

train.shape,test.shape       ########  train and test

#separate labels from other features
train_labels=train["Member type"].copy();
train=train.drop("Member type", axis=1)

test_labels=test["Member type"].copy()
test=test.drop("Member type",axis=1)

from sklearn.naive_bayes import GaussianNB   ##### guassinan
model=GaussianNB()

model.fit(train,train_labels)
predicted=model.predict(test)

predicted.shape

predicted

count=0                 ### accuracy
for l in range(len(predicted)):  
    if(predicted[l]==test_labels.iloc[l]):
        count=count+1

count

print(count/(len(predicted)))

from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score

clas=tree.DecisionTreeClassifier();
clas.fit(train,train_labels)
pred=clas.predict(test)
accu=accuracy_score(test_labels,pred)
accu

corr=data.corr();
sns.heatmap(corr,annot=True)

