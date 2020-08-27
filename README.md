# my_public_workspace

Input spreadsheet format:

CRITICAL PATH METHOD
#
#    input is an Excel spreadsheet (sheet1)
#    column 1: Activity name/description
#    column 2: Index associated with the activity (1,2,3....)
#    column 3: Activity duration
#    column 4: Index of that activity that this activity depends on            
#    column 5: ....
#    column n: One index per column. 0 if no dependency
#    see sample input excel file: cpm_data_1.xlsx
#    The spreadsheet is based on an example from Chapter 5 of
#    "Matching Supply with Demand" - G. Cachon and C. Terwiesch, 3rd Ed.

Required packages:

numpy, pandas and xlrd. 
To install these packages:
pip install pandas
pip install numpy
pip install xlrd

Program execution:
python project_cpm.py

Sample output:

enter Excel filename: cpm_data_1.xlsx 
###   CRITICAL PATH METHOD   ###
 
Total number of Activities: 10
Project duration: 57
 
List of Activities on the critical path:
 
a1 - a2 - a4 - a5 - a8 - a9 - a10
 
List of Activities with non-zero slack time:
 
Activity: a3  has slack time: 4
Activity: a6  has slack time: 2
Activity: a7  has slack time: 6
