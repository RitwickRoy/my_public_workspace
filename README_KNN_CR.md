# my_public_workspace

Input spreadsheet format:

KNearest Neighbor classificier/Regressor
#
#    input is an Excel spreadsheet (Sheet1)
#    column 1: Feature 1
#    column 2: Feature 2
#    column 3: .....       
#    column n-1: Feature n-1
#    column n: Target
#              For classifier the Target variable is a categorical variable
#              For regressor the Target variable is a continuous variable
#    see sample input excel file: Iris.xlsx
#    User input choices:
#            Operation: Classification/Regression task
#            data scaler: 0: no scaling
#                         1: Min-Max normalized scaling
#                         2: Standard Scaler
#            Ratio to split data in training and test set

Required packages:

numpy, pandas and xlrd. 
To install these packages:
pip install pandas
pip install numpy
pip install xlrd

Program execution:
python KNN_CR.py

Sample output:

Classification (0)/ Regression(1):  0

enter Excel filename: Iris.xlsx
data-scaling option:
 0 -  no scaling
 1 -  min-max scaling
 2 -  standard scaling

data scaling option: 2

Split Ratio for Training and Test Sets: 0.25

Number of Nearest Neighbors to use for predicition: 5

#### Validating against Test set ####


Accuracy:0.892
Support:     37

***Confusion Matrix***

[[12  0  0]
 [ 0 10  4]
 [ 0  0 11]]

    Label         Precision     Recall    F1-score   Support
    Iris-setosa       1.00       1.00       1.00       12
Iris-versicolor       1.00       0.71       0.83       14
 Iris-virginica       0.73       1.00       0.85       11
