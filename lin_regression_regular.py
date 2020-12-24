# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:01:40 2020

@author: Ritwick
"""
#
#    LINEAR REGRESSION 
#
#    input is an Excel spreadsheet (sheet1)
#    column 1: X (independent variable)
#    column 2: Y1
#    column 3: Y2
#    column 4: ....           
#    column 5: ....
#    column n: Yn
#    see sample input excel file: reg_data.xlsx
#    
#
import pandas as pd
import numpy as np
import random
import statistics

#
class LINREG:
    def __init__(self,df):
        self.df = df   
        self.nVar = self.df.shape[0]     #  Number of independent variables
        self.ScaleOpt = 1
        self.columns = list(self.df)
        self.nFVec = len(self.columns)-1
        self.Feature_data = np.zeros((self.nVar,self.nFVec+1))
        self.ScaleParam1 = [0.0 for i in range (self.nFVec)]
        self.ScaleParam2 = [0.0 for i in range (self.nFVec)]
        self.Y = np.zeros(self.nVar)
        self.LHS = np.zeros((self.nFVec+1,self.nFVec+1))
        self.RHS = np.zeros(self.nFVec+1)
        self.Coef = np.zeros(self.nFVec+1)
        self.randarr = random.sample(range(0,self.nVar),self.nVar)
        self.XTrain = np.empty([self.nVar,self.nFVec+1])
        self.yTrain = []
        self.XTest = np.empty([self.nVar,self.nFVec+1])
        self.yTest = []
        self.SplitRatio = 1.0
        self.nvarTrain = 1
        self.nvarTest = 0
        self.SSE = 0.0
        self.SSR = 0.0
        self.SST = 0.0
        self.R2  = 0.0
        self.meanytest = 0.0
        self.RegParam = 0.0
#
#   Input polynomial degree and data-scaling option
#   0 -  no scaling
#   1 -  min-max scaling
#   2 -  standard scaling
#
    def Read_options(self):
        print("data-scaling option:")
        print(" 0 -  no scaling")
        print(" 1 -  min-max scaling")
        print(" 2 -  standard scaling")
        txt = input("data scaling option: ")
        self.ScaleOpt = int(txt)
        str = input("Ridge Regularization parameter: ")
        self.RegParam = float(str)
#
#   Read split Ratio
#
    def ReadSplitRatio(self):
        u_str = input('Split Ratio for Training and Test Sets: ')
        self.SplitRatio = float(u_str)
#     
#   Compute Scaling parameter for each feature vector
#
    def ScaleParams(self):
        for iC in range (self.nFVec):
            if self.ScaleOpt == 2:
                self.ScaleParam1[iC] = self.df[self.columns[iC]].mean()
                self.ScaleParam2[iC] = self.df[self.columns[iC]].std()
            elif self.ScaleOpt == 1:
                self.ScaleParam1[iC] = self.df[self.columns[iC]].min()
                self.ScaleParam2[iC] = self.df[self.columns[iC]].max()
#
#  Setup every term that contribute the X matrix
#        
    def SetupTrainingdataLin (self):
#
        if self.ScaleOpt == 0:
            for i in range (self.nVar):
                for j in range (self.nFVec):
                    self.Feature_data[i][j] = self.df.loc[i,self.columns[j]]
                self.Feature_data[i][self.nFVec] = 1.0
        elif self.ScaleOpt == 1:
            for i in range (self.nVar):
                for j in range (self.nFVec):
                    d_max_min = self.ScaleParam2[j]-self.ScaleParam1[j]
                    self.Feature_data[i][j] = (self.df.loc[i,self.columns[j]]-self.ScaleParam1[j])/d_max_min
                self.Feature_data[i][self.nFVec] = 1.0
        elif self.ScaleOpt == 2:
            for i in range (self.nVar):
                for j in range (self.nFVec):
                    self.Feature_data[i][j] = (self.df.loc[i,self.columns[j]]-self.ScaleParam1[j])/self.ScaleParam2[j]
                self.Feature_data[i][self.nFVec] = 1.0
#
        for i in range(self.nVar):
            self.Y[i] = self.df.loc[i,self.columns[self.nFVec]]

#
#   Split the Target and Feature Vectors into Training and Test sets
#
    def TrainTestSplit(self):
        self.nVarTest = int(self.nVar*self.SplitRatio)
        self.nVarTrain = self.nVar-self.nVarTest
        for i in range(self.nVarTrain):
            loci = self.randarr[i]
            self.yTrain.append(self.Y[loci])
            for j in range(self.nFVec+1):
                self.XTrain[i,j] = self.Feature_data[loci,j]
#
        for i in range(self.nVarTrain,self.nVar):
            loci = self.randarr[i]
            self.yTest.append(self.Y[loci])
            for j in range(self.nFVec+1):
                self.XTest[i-self.nVarTrain,j] = self.Feature_data[loci,j]
#
        self.meanYtest = statistics.mean(self.yTest)

#
#   Compute X_T.X and X_T.Y 
#
    def LHS_RHS(self):
        for i in range(self.nFVec+1):
            for j in range(self.nFVec+1):
                for k in range(self.nVarTrain):
                    self.LHS[i][j] = self.LHS[i][j] + self.XTrain[k][i]*self.XTrain[k][j]
            self.LHS[i][i] += self.RegParam
#
        for i in range(self.nFVec+1):
            for j in range(self.nVarTrain):
                self.RHS[i] = self.RHS[i] + self.XTrain[j][i]*self.yTrain[j]
#
#   Solve system of equations for coefficients
#
    def Coef_Solve(self):
        self.Coef = np.linalg.solve(self.LHS,self.RHS)
#
#   Compute SSE, SSR and R2
#
    def RegError(self):
        self.SSE = 0.0
        self.SSR = 0.0
        for i in range(self.nVarTest):
            val = 0.0
            for j in range(self.nFVec+1):
                val += self.Coef[j]*self.XTest[i][j]
            self.SSE += (self.yTest[i]-val)**2
            self.SSR += (self.meanYtest-val)**2
        self.SSE = self.SSE / self.nVarTest
        self.SSR = self.SSR / self.nVarTest
        self.SST = self.SSR + self.SSE
        self.R2 = self.SSR / self.SST
#
#   Output from Regression
#
    def RegOutput(self):
        print()
        print("*** Linear Regression ***")
        print()
        print("Number of Independent Variables :",self.nFVec )
        print("Number of data points           :",self.nVar)
        print("Scaling option                  :",self.ScaleOpt)
        print()
        print("Coefficients :")
        print()
        coeff = []
        for j in range(self.nFVec):
            print(self.columns[j],"   :",self.Coef[j])
        print("Intercept :",self.Coef[self.nFVec])
        print()      
        print("Standard Error :",np.sqrt(self.SSE/(self.nVar-2)))
        print("SSE            :",self.SSE)
        print("SSR            :",self.SSR)
        print("SST            :",self.SST)
        print("R2             :",self.R2)
        
#
#  Read input Excel file
#

u_str = input('enter Excel filename: ')
df = pd.read_excel(u_str, sheet_name = 'Sheet1')  

#
#  instance the LINREG class with the input dataFrame
#
linreg_do = LINREG(df)
linreg_do.Read_options()
linreg_do.ScaleParams()
linreg_do.ReadSplitRatio()
linreg_do.SetupTrainingdataLin()
linreg_do.TrainTestSplit()
linreg_do.LHS_RHS()
linreg_do.Coef_Solve()
linreg_do.RegError()
linreg_do.RegOutput()
    
