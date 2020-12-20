# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:46:07 2020

@author: Ritwick
"""
#
#  KNN classifier/Regressor
#  Notes: This is a basic implementation that is good for small size datasets.
#  1. The data-set is split in Training and Test data
#  2. predictions are checked against the Target Labels/values
#  3. Classification assumes categorical target variable in the final column
#  4. Regression assumed continuous target variable in the final column
#  Future imporvements: Use binning/bucket sorting to speed up neighbor serach
#
import pandas as pd
import numpy as np
import seaborn as sns
import random
import statistics
#
class knn:
    def __init__(self,df,ClassRegInp):
        self.df = df
        self.nrows = self.df.shape[0]
        self.ncols = self.df.shape[1]
        self.nneighbors = 1
        self.col_names = df.columns
        self.TargetV = self.df[self.col_names[self.ncols-1]]
        self.FeatureV = self.df.drop(self.col_names[self.ncols-1],axis=1)
        self.ScaleOpt = 1
        self.ScaleParam1 = [0.0 for i in range (self.ncols)]
        self.ScaleParam2 = [0.0 for i in range (self.ncols)]
        self.v1 = np.zeros(self.ncols)
        self.v2 = np.zeros(self.ncols)
        self.distVal = 0.0
        self.dist = []
        self.probabilty = 0.0
        self.Label_p = ''
        self.TargetLabel = ''
        self.Label_pr = 0.0
        self.TargetLabelr = 0.0
        self.randarr = random.sample(range(0,self.nrows),self.nrows)
        self.XTrain = np.empty([self.nrows,self.ncols-1])
        self.yTrain = []
        self.XTest = np.empty([self.nrows,self.ncols-1])
        self.yTest = []
        self.SplitRatio = 1.0
        self.nrowsTrain = 1
        self.nrowsTest = 0
        self.score = 0
#
        self.ClassRegFlag = ClassRegInp
        if ClassRegInp == 0:
            self.TargetLabelsUnique = self.TargetV.unique() 
            self.NUniqueLabels = self.TargetV.nunique() 
            self.CM = np.zeros([self.NUniqueLabels,self.NUniqueLabels],dtype=int)
        elif ClassRegInp == 1:
            self.SSE = 0.0
            self.SSR = 0.0
            self.SST = 0.0
            self.R2  = 0.0
            self.meanYtest = 0.0
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
#     
#   Compute Scaling parameter for each feature vector
#
    def ScaleParams(self):
        for iC in range (self.ncols-1):
            if self.ScaleOpt == 2:
                self.ScaleParam1[iC] = self.FeatureV[self.col_names[iC]].mean()
                self.ScaleParam2[iC] = self.FeatureV[self.col_names[iC]].std()
            elif self.ScaleOpt == 1:
                self.ScaleParam1[iC] = self.FeatureV[self.col_names[iC]].min()
                self.ScaleParam2[iC] = self.FeatureV[self.col_names[iC]].max()
#
#   Transform feature data based on the specified scaling option
#        
    def ScaleFeaturedata (self):
#
        if self.ScaleOpt == 1:
            for i in range (self.nrows):
                for j in range (self.ncols-1):
                    d_max_min = self.ScaleParam2[j]-self.ScaleParam1[j]
                    self.FeatureV.iloc[i,j] = (self.FeatureV.loc[i,self.col_names[j]]-self.ScaleParam1[j])/d_max_min
        elif self.ScaleOpt == 2:
            for i in range (self.nrows):
                for j in range (self.ncols-1):
                    self.FeatureV.iloc[i,j] = (self.FeatureV.loc[i,self.col_names[j]]-self.ScaleParam1[j])/self.ScaleParam2[j]
#
#   Split the Target and Feature Vectors into Training and Test sets
#
    def TrainTestSplit(self):
        self.nrowsTest = int(self.nrows*self.SplitRatio)
        self.nrowsTrain = self.nrows-self.nrowsTest
        for i in range(self.nrowsTrain):
            loci = self.randarr[i]
            self.yTrain.append(self.TargetV.iloc[loci])
            for j in range(self.ncols-1):
                self.XTrain[i,j] = self.FeatureV.iloc[loci,j]
#
        for i in range(self.nrowsTrain,self.nrows):
            loci = self.randarr[i]
            self.yTest.append(self.TargetV.iloc[loci])
            for j in range(self.ncols-1):
                self.XTest[i-self.nrowsTrain,j] = self.FeatureV.iloc[loci,j]
#
        if self.ClassRegFlag == 1:
            self.meanYtest = statistics.mean(self.yTest)
#
#   Read split Ratio
#
    def ReadSplitRatio(self):
        u_str = input('Split Ratio for Training and Test Sets: ')
        self.SplitRatio = float(u_str)
#
#   Compute square of Euclidean distance between two feature vectors
#
    def EuclidDist(self):
        distV = 0.0
        for i in range (self.ncols-1):
            distV += (self.v1[i] - self.v2[i])**2
        self.distVal = distV
#
#    Compute Euclid distance between new Feature vector and training  dataset
#
    def Dist(self):
        self.dist = []
        for i in range(self.nrowsTrain):
            for j in range(self.ncols-1):
                self.v2[j] = self.XTrain[i,j]
            self.EuclidDist()
            self.dist.append(self.distVal)
#
#    Predict Target for New Feature vector
#
    def KNearestN(self):
        df_L = pd.DataFrame({'dist':self.dist,'Labels':self.yTrain})
        df_L.sort_values('dist',ascending=True,inplace=True)
        df_s = df_L.iloc[:self.nneighbors,:]
        df_t = df_s.groupby('Labels',as_index=False).count()
        df_ts = df_t.sort_values('dist',ascending=False)
        self.Label_p = df_ts.iloc[0,0]
        self.probability = df_ts.iloc[0,1]/df_ts.iloc[:,1].sum()
#
#    Predict Regression Target for New Feature vector
#
    def KNearestNR(self):
        df_L = pd.DataFrame({'dist':self.dist,'Labels':self.yTrain})
        df_L.sort_values('dist',ascending=True,inplace=True)
        df_s = df_L.iloc[:self.nneighbors,:]
        self.Label_pr = df_s.loc[:,'Labels'].mean()
#
#   Read number of neighbors
#
    def ReadNumNeighbors(self):
        u_str = input('Number of Nearest Neighbors to use for predicition: ')
        self.nneighbors = int(u_str)
#
#   Driver method for classifier: Test prediction for new feature Vector
#
    def TestNewFV(self):
        for i in range (self.nrowsTest):
            for j in range(self.ncols-1):
                self.v1[j] = self.XTest[i,j]
            self.TargetLabel = self.yTest[i]
            self.Dist()
            self.KNearestN()
            self.CheckPredLabel()
#
#   Driver method for regressor: Test prediction for new feature Vector (Regression)
#
    def TestNewFVR(self):
        for i in range (self.nrowsTest):
            for j in range(self.ncols-1):
                self.v1[j] = self.XTest[i,j]
            self.TargetLabelr = self.yTest[i]
            self.Dist()
            self.KNearestNR()
            self.CheckPredLabelR()
#
#   Check predicted Vs Test Label and compute confusion matrix
#            
    def CheckPredLabel(self):
        if self.Label_p == self.TargetLabel:
           self.score += 1
#           
        if self.Label_p == self.TargetLabel:
           for i in range(self.NUniqueLabels):
               if self.TargetLabel == self.TargetLabelsUnique[i]:
                   self.CM[i][i] += 1
        elif self.Label_p != self.TargetLabel:
           for i in range(self.NUniqueLabels):
               if self.TargetLabel == self.TargetLabelsUnique[i]:
                   for j in range(self.NUniqueLabels):
                       if self.Label_p == self.TargetLabelsUnique[j]:
                           self.CM[i][j] += 1

#
#   squared error between predicted Vs Test Label
#            
    def CheckPredLabelR(self):
        self.score += (self.Label_pr - self.TargetLabelr)**2
        self.SSE = self.SSE + (self.TargetLabelr-self.Label_pr)**2
        self.SSR = self.SSR + (self.meanYtest-self.Label_pr)**2
#
#   Output Confusion Mtrix and Classification Report
#        
    def Output(self):
        self.score = self.score/self.nrowsTest
        print(f'Accuracy:{self.score:5.3f}')
        print(f'Support:{self.nrowsTest:7d}')
        print()
        print('***Confusion Matrix***')
        print()
        print(self.CM)
        print()
        print('    Label         Precision     Recall    F1-score   Support')
        for i in range (self.NUniqueLabels):
            p_num = self.CM[i][i]
            p_den = 0.0
            r_num = self.CM[i][i]
            r_den = 0.0
            support = 0
            for j in range (self.NUniqueLabels):
                p_den += self.CM[j][i]
                r_den += self.CM[i][j]
                support += self.CM[i][j]
            precision = p_num/p_den
            recall = r_num/r_den
            f1_score = (2.0*precision*recall)/(precision+recall)
            print(f'{self.TargetLabelsUnique[i]:>15}','%10.2f %10.2f %10.2f  %7d'%(precision,recall,f1_score,support))
#
#   Output mean squared error
#        
    def OutputR(self):
        self.SSE = self.SSE/self.nrowsTest
        self.SSR = self.SSR/self.nrowsTest
        self.SST = self.SSR + self.SSE
        self.R2 = self.SSR/self.SST    
        print('MSSE :',self.SSE)
        print('MSSR :',self.SSR) 
        print('MSST :',self.SST)
        print('R2  :',self.R2)
#
#  Read Data-Set
#
u_str = input('Classification (0)/ Regression(1):  ')
ClassRegFlag = int(u_str)
u_str = input('enter Excel filename: ')
df = pd.read_excel(u_str, sheet_name = 'Sheet1')  
#
knn_do = knn(df,ClassRegFlag)
knn_do.Read_options()
knn_do.ScaleParams()
knn_do.ScaleFeaturedata()
knn_do.ReadSplitRatio()
knn_do.ReadNumNeighbors()
knn_do.TrainTestSplit()
print()
print('#### Validating against Test set ####')
print()
if ClassRegFlag == 0:
    knn_do.TestNewFV()
    print()
    knn_do.Output()
elif ClassRegFlag == 1:
    knn_do.TestNewFVR()
    print()
    knn_do.OutputR()    

