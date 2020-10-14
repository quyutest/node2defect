# -*- coding:utf-8 -*-
'''
Created on 2018年9月21日

@author: Yu Qu
'''

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import copy

class Smote(object):
    '''
    classdocs
    '''


    def __init__(self,samples,N=50,k=5,r=2):
        '''
        Constructor
        '''
        self.samples=copy.deepcopy(samples)
#         print(samples)
        self.T,self.numattrs=self.samples.shape
        self.N=N
        self.k=k
        self.r=r#这个r是计算Minkowski距离时的幂指数
        self.newindex=0
        
            
        
    def generate_synthetic_points(self):
        if(self.N<100):
            np.random.shuffle(self.samples)
            self.T=int(self.N*self.T/100)
            self.samples=self.samples[0:self.T,:]
            self.N=100
        if(self.T<self.k):
            self.k=self.T-1
        
        N=int(self.N/100)
        self.synthetic = np.zeros((self.T * N, self.numattrs))       
        neighbors=NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree', p=self.r).fit(self.samples)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape((1,-1)),return_distance=False)[0]#Finds the K-neighbors of a point.
            self._populate(N,i,nnarray)
        return self.synthetic
    
    def _populate(self,N,i,nnarray):
        for j in range(N):
            attrs=[]
            nn=random.randint(0,self.k-1)
            for attr in range(self.numattrs):
                diff = self.samples[nnarray[nn]][attr] - self.samples[i][attr]
                gap = random.uniform(0,1)
                attrs.append(self.samples[i][attr] + gap*diff)
            self.synthetic[self.newindex]=attrs
            self.newindex+=1
            
# a=np.array([[1,2,3],[4,5,6],[2,3,1],[2,1,2],[2,3,4],[2,3,4]])
# 
# smote=Smote(a,N=50)
# synthetic_points=smote.generate_synthetic_points()
# # print synthetic_points
# 
# smote=Smote(a,N=100)
# synthetic_points=smote.generate_synthetic_points()
# print synthetic_points
        
        