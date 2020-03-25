import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing


### Load data from csv file
df = pd.read_csv('loan_train.csv')

print(df.head()) # by defualt, show the first 5 rows
print(df.shape) # prints numbers of row and column

### Convert to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
print(df.head()) # by defualt, show the first 5 rows

print(df['loan_status'].value_counts()) # show numbers of PAIDOFF and COLLECTION

### Convert categorical features to numerical values
print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)) # show numbers of PAIDOFF and COLLECTION by male and female
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True) # Convert male to 0 and female to 1
print(df.head())

print(df.groupby(['education'])['loan_status'].value_counts(normalize=True)) # show numbers of PAIDOFF and COLLECTION by education

### Pre-processing: Feature selection/extraction
df['dayofweek'] = df['effective_date'].dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)

### Define feature sets
X = Feature
y = df['loan_status'].values

### Normalize data
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

# Split testing and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

### Approach 1: K Nearest Neighbor (kNN)

# 1. Import KNN library
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 2. Find the best k
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = [];
best_k = 1
best_acc = -1
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
    if mean_acc[n - 1] > best_acc:
        best_acc = mean_acc[n - 1]
        best_k = n

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

print(mean_acc, "Best k: " + str(best_k))

# 3. Train model w the best k
neigh = KNeighborsClassifier(n_neighbors = best_k).fit(X_train,y_train)

yhat = neigh.predict(X_test)

# 4. Evaluation
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

### Approach 2: Decision Tree
from sklearn.tree import DecisionTreeClassifier

DT_model = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
DT_model.fit(X_train,y_train)

yhat = DT_model.predict(X_test)
print(yhat)

### Approach 3: SVM
from sklearn import svm

SVM_model = svm.SVC()
SVM_model.fit(X_train, y_train)

yhat = SVM_model.predict(X_test)
print(yhat)

### Approach 4: Logistic Regression
from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression(C=0.01).fit(X_train,y_train)

yhat = LR_model.predict(X_test)
print(yhat)

### Model evaluation using test set

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

# 1. Load test set
test_df = pd.read_csv('loan_test.csv')

# 2. Pre-processing
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace=True)

test_X = preprocessing.StandardScaler().fit(test_Feature).transform(test_Feature)
test_y = test_df['loan_status'].values
print(test_X[0:5], test_y[0:5])

# 3. kNN
knn_yhat = neigh.predict(test_X)
print("KNN Jaccard index: %.2f" % jaccard_similarity_score(test_y, knn_yhat))
print("KNN F1-score: %.2f" % f1_score(test_y, knn_yhat, average='weighted') )

# 4. Decision Tree
DT_yhat = DT_model.predict(test_X)
print("DT Jaccard index: %.2f" % jaccard_similarity_score(test_y, DT_yhat))
print("DT F1-score: %.2f" % f1_score(test_y, DT_yhat, average='weighted') )

# 5. SVM Model
SVM_yhat = SVM_model.predict(test_X)
print("SVM Jaccard index: %.2f" % jaccard_similarity_score(test_y, SVM_yhat))
print("SVM F1-score: %.2f" % f1_score(test_y, SVM_yhat, average='weighted') )

# 6. Logistic Regression Model
LR_yhat = LR_model.predict(test_X)
LR_yhat_prob = LR_model.predict_proba(test_X)
print("LR Jaccard index: %.2f" % jaccard_similarity_score(test_y, LR_yhat))
print("LR F1-score: %.2f" % f1_score(test_y, LR_yhat, average='weighted') )
print("LR LogLoss: %.2f" % log_loss(test_y, LR_yhat_prob))





