# -*- coding:utf-8 -*-
'''
Created on 2018年9月23日

@author: Yu Qu
'''
import numpy as np
from SMOTUNED.smote import Smote

def smote_wrapper(new, train_data, train_label):
    data_t, data_f, label_t, label_f = [], [], [], []
    N = train_label.shape[0]
    for i in range(N):
        if train_label[i] == 1:
            data_t.append(train_data[i])
            label_t.append(train_label[i])
        if train_label[i] == 0:
            data_f.append(train_data[i])
            label_f.append(train_label[i])
    data_t_np=np.array(data_t)
#     print(data_t)
    smote=Smote(data_t_np,k=new[0], N=new[1], r=new[2])
    synthetic_points=smote.generate_synthetic_points()
    data_t_np=np.vstack((data_t_np,synthetic_points))
    label_add = np.ones(synthetic_points.shape[0])
    label_t.extend(label_add)
    
    data_new=np.vstack((data_t_np,np.array(data_f)))
    label_new=np.append(label_t, label_f, axis=0)
    
    return data_new,label_new