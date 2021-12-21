import pandas as pd
import seaborn as sns
from scipy import stats
import math
from sklearn.datasets import load_iris

data = load_iris()

df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
df['target'] = data['target']

y = df['target']
x = df.drop(['target'],axis=1)

# No. of features and their types

print("Number of features: ",len(df.columns) - 1)

x.dtypes

y.value_counts()

# Display summary for each column

x.describe() #using API

#min, max manually
for i in df:
  if i != "class":
    print(i)
    print("min", df[i].min())
    print("max", df[i].max())
    print("range: ", df[i].max() - df[i].min(), "\n")

#mean
def calculate_mean(df, col):
    sum = 0.0
    count = 0
    for i in df[col]:
        if(type(i) == str): 
            return
        sum= sum + i
        count+=1
    return sum/count

for i in df:
    if(i != "class"):
        print("Mean of ", i, " is ",calculate_mean(df, i))

#Standard deviation
def calculate_std(df, col):
    mean = calculate_mean(df, col)
    dif = 0
    count = 0
    for i in df[col]:
        dif += (i - mean)**2
        count += 1
    return math.sqrt(dif / count)

for i in df:
    if(i != "class"):
        print("Standard Deviation of ", i, " is ",calculate_std(df, i))


# # Data Visualization

import matplotlib.pyplot as plt
plt.hist(x['sepal length (cm)'],bins=15,color='green')
plt.show()

plt.hist(x['sepal width (cm)'],orientation='vertical')
plt.show()

plt.hist(x['petal length (cm)'])
plt.title('Variations in petal length')
plt.xlabel('Petal length')
plt.ylabel('Frequency')
plt.show()

df.plot.hist(subplots=True, legend=True)

x.boxplot()




