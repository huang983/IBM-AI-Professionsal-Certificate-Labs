# Loan Repayment Prediction Using Machine Learning
Use an ML model to determine if the borrower has the ability to repay its loan

# Dataset:
    1. Loan_train.csv
    2. Loan_test.csv
    
Field          | 
    |-----------------------|
    | Loan_status           |
    | Principal             |
    | Terms                 |
    | Effective Date        |
    | Due_date              |
    | Age                   |
    | Education             |
    | Gender                |

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
