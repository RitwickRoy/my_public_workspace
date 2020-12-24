# my_public_workspace

Input spreadsheet format:

KMeans clustering
#
#    input is an Excel spreadsheet (Sheet1)
#    column 1: X (independent variable)
#    column 2: Y1
#    column 3: .....       
#    column n: Yn
#
#    see sample input excel file: california_housing_train_clean.xlsx
#    User input choices:
#            data scaler: 0: no scaling
#                         1: Min-Max normalized scaling
#                         2: Standard Scaler
#            Ridge Regularization parameter
#            Train-Test set split ratio
#
Required packages:

numpy, pandas and xlrd. 
To install these packages:
pip install pandas
pip install numpy
pip install xlrd

Program execution:
python lin_regression_regular.py

Sample output:

enter Excel filename: california_housing_train_clean.xlsx
data-scaling option:
 0 -  no scaling
 1 -  min-max scaling
 2 -  standard scaling

data scaling option: 2

Ridge Regularization parameter: 1.2

Split Ratio for Training and Test Sets: 0.2

*** Linear Regression ***

Number of Independent Variables : 6
Number of data points           : 17000
Scaling option                  : 2

Coefficients :

housing_median_age    : 23831.602157379773
total_rooms    : -42339.85860261926
total_bedrooms    : 42365.14744496615
population    : -38303.78340380469
households    : 45646.288128425185
median_income    : 91431.3516277432
Intercept : 207470.86707797152

Standard Error : 567.2922904215405
SSE            : 5470305586.033654
SSR            : 7584339822.329967
SST            : 13054645408.36362
R2             : 0.5809686579055579


