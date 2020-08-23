# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:01:40 2020

@author: Ritwick
"""
#
#    CRITICAL PATH METHOD
#
import pandas as pd
import numpy as np
import sys
#
class CPM:
    def __init__(self,df):
        self.df = df   
        self.nAct = self.df.shape[0]     #  Number of Activities
        self.df['nAdjacency']=0  
        self.df['Status'] = 1
        self.df['EST'] = 0
        self.df['EFT'] = 0
        self.df['LST'] = 0
        self.df['LFT'] = sys.maxsize 
        self.df['Slack Time'] = 0   
        self.columns = list(self.df)
        nDependencyMax = self.df['Dependency Length'].max()
        self.DependencyList = np.zeros((self.nAct,nDependencyMax),dtype=int)  
        self.StackLength = 2*self.nAct   # Size the stack for recussion, may need to be increased for larger problems
        self.stack =  [0 for i in range (self.StackLength)]   
#     
#   Helper methods that are used in the recursion
#
    def EFT_EST(self, iLoc):
        self.df.loc[iLoc,'EFT'] = self.df.loc[iLoc,'EST']+self.df.loc[iLoc,'Activity Time']
#
    def LST_LFT(self,iLoc):
        self.df.loc[iLoc,'LST'] = self.df.loc[iLoc,'LFT']-self.df.loc[iLoc,'Activity Time']
#
    def EST_max_EFT(self, iAdj, iLoc):
        self.df.loc[iAdj-1,'EST'] = max(self.df.loc[iAdj-1,'EST'],self.df.loc[iLoc,'EFT'])
#
    def LFT_min_LST(self, iAdj, iLoc):
        self.df.loc[iAdj-1,'LFT'] = min(self.df.loc[iAdj-1,'LFT'],self.df.loc[iLoc,'LST'])
#
    def CheckActivityTreated(self, iAdj, ADList, ADLength):
        iActTreated = 0
        for k in range (self.df.loc[iAdj-1,ADLength]):
            iDep = ADList[iAdj-1][k] - 1
            if self.df.loc[iDep,'Status'] == 3:
                iActTreated += 1
        return iActTreated
#
#
#   Extract the columns containing the dependencies
#   and store is a dependency list array
#
    def Build_DependencyList(self):
        columns = list(self.df)
        n = len(columns)
        iStart = 0
        iEnd = 0
        for i in range (n):
            if columns[i] == 'Dependency Length':
                iStart = i
            if columns[i] == 'nAdjacency':
                iEnd  = i
#
        colList = []
        for i in range (iStart+1, iEnd, 1):
            colList.append(columns[i])
#
#       Compute the length of adjacency list for each activity
#
        for i in range (self.nAct):
           j = 0
           for column in self.df[colList]:
               self.DependencyList[i][j] = self.df.loc[i,column]
               iC = self.df.loc[i,column]
               if iC > 0 :
                   self.df.loc[iC-1,'nAdjacency'] = self.df.loc[iC-1,'nAdjacency'] + 1
               j += 1
#
#
    def Build_AdjacencyList(self):
#
#       initialize the Adjacency list
#        
        nCount = np.zeros(self.nAct,dtype=int)      
        nAdjacencyMax = self.df['nAdjacency'].max()
        self.AdjacencyList = np.zeros((self.nAct,nAdjacencyMax),dtype=int)
#
#       Setup the Adjacency list
#
        for i in range (self.nAct):
            for j in range (self.df.loc[i,'Dependency Length']):
                iC = self.DependencyList[i][j]
                if iC > 0 :
                    k = nCount[iC-1]
                    self.AdjacencyList[iC-1][k] = i+1
                    nCount[iC-1] += 1
                j += 1
#
#
    def Forward_Sweep(self):
#
#       push the first activity into the stack
#
        iStkptr = 0
        self.stack[iStkptr] = self.df.loc[0, 'Activity Index']
        self.df.loc[0,'Status'] = 2
#
#       Use depth-first recursion to traverse the activity graph in the forward direction
#
        iStkptr += 1
        while iStkptr > 0:
            iStackTop = self.stack[iStkptr-1] - 1      # pop stack
            if self.df.loc[iStackTop,'Status'] == 2:
                self.EFT_EST(iStackTop)
                self.df.loc[iStackTop,'Status'] = 3
                iStkptr = iStkptr - 1
                for j in range(self.df.loc[iStackTop,'nAdjacency']):
                    iAdj = self.AdjacencyList[iStackTop][j]
                    self.EST_max_EFT(iAdj, iStackTop)
                    iAllDependencyTreated = self.CheckActivityTreated(iAdj, self.DependencyList,'Dependency Length')
                    if iAllDependencyTreated == self.df.loc[iAdj-1,'Dependency Length']:
                        if self.df.loc[iAdj-1,'Status'] == 1: 
                            self.stack[iStkptr] = iAdj
                            self.df.loc[iAdj-1,'Status'] = 2
                            iStkptr += 1
#
#
    def Reverse_Sweep(self):
#
#       Reinitialize the stack for reverse sweep
#
        self.df['Status'] = 1
        self.stack =  [0 for i in range (self.StackLength)]   
#
#       push the last activity into the stack
#
        iStkptr = 0
        self.stack[iStkptr] = self.df.loc[self.nAct-1, 'Activity Index']
        self.df.loc[self.nAct-1,'Status'] = 2
        self.df.loc[self.nAct-1,'LFT'] = self.df.loc[self.nAct-1,'EFT']
#
#       Use depth-first recursion to traverse the activity graph in the reverse direction
#
        iStkptr += 1
        while iStkptr > 0:
            iStackTop = self.stack[iStkptr-1] - 1      # pop stack
            if self.df.loc[iStackTop,'Status'] == 2:
                self.LST_LFT(iStackTop)
                self.df.loc[iStackTop,'Status'] = 3
                iStkptr = iStkptr - 1
                for j in range(self.df.loc[iStackTop,'Dependency Length']):
                    iAdj = self.DependencyList[iStackTop][j]
                    self.LFT_min_LST(iAdj, iStackTop)
                    iAllAdjacencyTreated = self.CheckActivityTreated(iAdj, self.AdjacencyList,'nAdjacency')
                    if iAllAdjacencyTreated == self.df.loc[iAdj-1,'nAdjacency']:
                        if self.df.loc[iAdj-1,'Status'] == 1: 
                            self.stack[iStkptr] = iAdj
                            self.df.loc[iAdj-1,'Status'] = 2
                            iStkptr += 1
#
#
    def Output(self):
#
#       Compute the slack time for each activity
#
        for i in range (self.nAct):
            self.df.loc[i,'Slack Time'] = self.df.loc[i,'LST'] - self.df.loc[i,'EST']
#
#       output the project duration, critical path and slack times
#
        print(' ')
        print("###   CRITICAL PATH METHOD   ###")
        print(' ')
        print('Total number of Activities:', self.nAct)
        print('Project duration:',self.df.loc[self.nAct-1,'EFT']) 
        print(' ')
        print('List of Activities on the critical path:')  
        critical_path = [] 
        for i in range (self.nAct):
            if self.df.loc[i,'Slack Time'] == 0:
                critical_path.append(self.df.loc[i,'Activity Name'])
#
        print(' ')
        print(' - '.join(str(value) for value in critical_path))
        print(' ')
        print('List of Activities with non-zero slack time:')
        print(' ')
        for i in range (self.nAct):
            if self.df.loc[i,'Slack Time'] > 0:
                print('Activity:',self.df.loc[i,'Activity Name'],' has slack time:',self.df.loc[i,'Slack Time'])
#
#
#  Read input Excel file
#
u_str = input('enter Excel filename: ')
df = pd.read_excel(u_str, sheet_name = 'Sheet1')  
#
#  instance the cpm class with the read data
#
cpm_do = CPM(df)
cpm_do.Build_DependencyList()
cpm_do.Build_AdjacencyList()
cpm_do.Forward_Sweep()
cpm_do.Reverse_Sweep()
cpm_do.Output()
        
    
