# -*- coding:utf-8 -*-
'''
Created on 2018年9月19日

@author: Yu Qu
'''
import random
from SMOTUNED.smote import Smote
from SMOTUNED.wrapper import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import copy
from SupportingTools.ClassifierOutput import *

from sklearn import metrics

class SMOTUNED(object):
    def __init__(self, n=10, cf=0.3, f=0.7):
        self.n = n  # Population Size: Frontier size in a generation
        self.cf = cf  # Crossover probability: Survival of the candidate
        self.f = f  # Differential weight: Mutation power
        self.k = 20  # Number of neighbors: [1,20]
        self.m = [50, 100, 200, 400, 800] # Number of synthetic examples to create. Expressed as percent of final training data
        self.r = 5  # Power parameter for the Minkowski distance metric: [0.1, 5]
        
    def DE(self, train_data, train_label, test_data, test_label):
        
        
        frontier = self.guessed()
        best = frontier[0]
        best_score = 0
        lives = 1  # Number of generations
        while lives > 0:
            lives = lives - 1
#             print "best="+str(best)
#             print "New Lives="+str(lives)
            tmp = []
            for i in range(len(frontier)):
                old = frontier[i]                
                index = random.sample(range(len(frontier)), 3)
                x, y, z = frontier[index[0]], frontier[index[1]], frontier[index[2]]
                new = list(old)
                for j in range(3):
                    if random.random() < self.cf:
                        if j == 0:
                            temp_1 = int(x[j] + self.f * (z[j] - y[j]))
                            if temp_1 >= 1 and temp_1 <= 20:
                                new[j] = temp_1
                        if j == 1:
#                             temp_2 = x[j] + self.f * (z[j] - y[j])
#                             if temp_2 > 0 and temp_2 <= 400:
#                                 new[j] = temp_2   
                            new[j] = random.choice(self.m)
                        if j == 2:
                            temp_3 = round(x[j] + self.f * (z[j] - y[j]), 1)
#                             if temp_3 >= 0.1 and temp_3 <= 5:
                            if temp_3 > 1 and temp_3 <= 5:
                                new[j] = temp_3
                
                new, is_, score = self.better(new, old, train_data, train_label, test_data, test_label)
                tmp.append(new)
                
                if(new==best):
                    continue
                
                print(best_score)
                print(score)
                     
                if(score>best_score):
                    best_score=score
                    best=new
                    lives=lives+1
                    print("The best score: "+str(best_score))
                    print("Lives="+str(lives))
                
                
                
#                 t, is_, score = self.better(new, best, train_data, train_label, test_data, test_label)
#                 if is_:
#                     print "new="+str(t)
#                     print "best="+str(best)
#                          
#                     best = t
#                     lives += 1
#                     print "The best score: "+str(score)
#                     print "Lives="+str(lives)
            frontier = tmp
            
            print(frontier)
            print("*****Complete One Generation!****")
            
            
        return best[0], best[1], best[2]
    
    
    # initialize frontier
    def guessed(self):
        frontier = []
        for i in range(self.n):
            k = random.randint(1, self.k+1)
            m = random.choice(self.m)
            r = float(random.randint(10, self.r*10+1))/10
            frontier.append([k, m, r])
        return frontier
    
    @staticmethod
    # fitness Function
    def better(new, old, train_data, train_label, test_data, test_label):
        auc_new_all=0
        auc_old_all=0
        for i in range(10):
        
            data_new,label_new=smote_wrapper(new, train_data, train_label)
            data_old,label_old=smote_wrapper(old, train_data, train_label)
            
            
            predprob_auc_new,predprob_new,precision_new,recall_new,fmeasure_new,auc_new=classifier_output(data_new,label_new,test_data,test_label,grid_sear=False)
#             auc_new_all=auc_new_all+auc_new
            auc_new_all=auc_new_all+fmeasure_new
            
            predprob_auc_old,predprob_old,precision_old,recall_old,fmeasure_old,auc_old=classifier_output(data_old,label_old,test_data,test_label,grid_sear=False)
#             auc_old_all=auc_old_all+auc_old
            auc_old_all=auc_old_all+fmeasure_old
            
        auc_new_all=auc_new_all/10
        auc_old_all=auc_old_all/10
        print("new="+str(new))
        print("new score="+str(auc_new_all))#Can be changed to other metrics.
        print("old="+str(old))
        print("old score="+str(auc_old_all))
        metric_new=auc_new_all
        metric_old=auc_old_all
        
        if metric_new > metric_old:
            return new, True, metric_new
        else:
            return old, False, metric_old