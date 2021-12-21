import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pima = pd.read_csv('./diabetes.csv')
pima

pima.describe()

pima.info()

pima['Outcome'].value_counts()


# There are no NULL values. But 0 value is there. We will replace 0 values with mean of respecetive column

pima[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = pima[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(pima.isnull().sum())

pima['Glucose'].fillna(pima['Glucose'].mean(), inplace = True)
pima['BloodPressure'].fillna(pima['BloodPressure'].mean(), inplace = True)
pima['SkinThickness'].fillna(pima['SkinThickness'].mean(), inplace = True)
pima['Insulin'].fillna(pima['Insulin'].mean(), inplace = True)
pima['BMI'].fillna(pima['BMI'].mean(), inplace = True)

print(pima.isnull().sum())


# Results of Data Cleaning:

pima.describe().T

# Distribution of Data

pima.hist(figsize=(20,16), grid=True)

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


# Splitting Data into Training and Testing sets

X = pima.drop('Outcome', axis  = 1)
y = pima['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 17)

print(X_train.shape)
print(X_test.shape)
print(y_train.size)
print(y_test.size)


# Building a model using Naive Bayes Classifier

nbModel = GaussianNB()

nbModel.fit(X_train, y_train)

nb_y_pred = nbModel.predict(X_test)

# Evaluating the model using Confusion Matrix

nbConfusion = metrics.confusion_matrix(y_test, nb_y_pred)
nbConfusion

ylabel = ["Actual [Non-Diabetic]","Actual [Diabetic]"]
xlabel = ["Pred [Non-Diabetic]","Pred [Diabetic]"]
plt.figure(figsize=(15,6))
sns.heatmap(nbConfusion, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)

print('Accuracy of Naive Bayes Classifier is: ', nbModel.score(X_test,y_test) * 100,'%')

print(classification_report(y_test, nb_y_pred))

# Here, 0 indicates No Diabetes, 1 indicates Diabetes

TP = nbConfusion[1, 1]
TN = nbConfusion[0, 0]
FP = nbConfusion[0, 1]
FN = nbConfusion[1, 0]

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
Specificity = TN / (TN + FP)

print("Precision: ", Precision)
print("Recall (Sensitivity): ", Recall)
print("Specificity: ", Specificity)

# F-measure

f1 = (2*Precision*Recall)/(Precision+Recall)
print("F1 Score: ", f1)
