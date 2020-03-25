# Loan Repayment Prediction Using Machine Learning
Use an ML model to determine if the borrower has the ability to repay its loan

# Dataset:
    1. Loan_train.csv
    2. Loan_test.csv
    
Field|
-----------------------|
Loan_status           |
Principal             |
Terms                 |
Effective Date        |
Due_date              |
Age                   |
Education             |
Gender                |

# Library:
    1. sklearn
    2. pandas
    3. numpy
    4. matplotlib
    
# Functions 

## Preprocessing:
 ```python
    from sklearn import preprocessing
    X = preprocessing.StandardScaler().fit(X).trasnform(X) 
```
## Split test and train
 ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

## Approach 1: KNN Algorithm

- Classifying cases bsed on their similarity to other cases
- Cases near each other are said to be neighbors

```buildoutcfg
1. Pick a value for K
2. Calculate the distancfe of unkown case from all cases
3. Select the K-observations in the training data that are nearest to the unkown data point
4. Predict the response of the unkown data point using the most popular response value from the K-nearest neighbors
```


## Evaluation Metrics
1. Jaccard index:
    * y: actual labels
    * y_hat: predicted labels
    * Accuracy = intersection(y, y_hat) / union(y, y_hat)
        
2. Confusion matrix:
    * Binary matrix
    * TP, FN, FP, TN
    * True labels represented by rows
    * Predicted labels represented by columns
    * Precision = TP / (TP + FP)
    * Recall = TP / (TP + FN) 
    * F1-score = 2 * (prc * rec) / (prc + rec)
    
3. Log loss:
    * Performance of a classifier where the predicted output is a probability value between 0 and 1
    