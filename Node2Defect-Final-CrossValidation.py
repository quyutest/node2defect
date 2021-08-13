# -*- coding:utf-8 -*-
'''
@author: Qu Yu
'''

import networkx as nx
import shutil

import random
import time
import os
import shutil
from SupportingTools.ReadFile import *
from SupportingTools.CDNAnalysis import *
from sklearn.model_selection import RepeatedKFold, train_test_split
import numpy
from scipy import stats
import shutil
from SMOTUNED.smote import Smote
from SMOTUNED.wrapper import *
from SupportingTools.ClassifierOutput import *
import warnings
import math
import datetime
from sklearn.linear_model import LinearRegression
from info_gain import info_gain
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler


def output_results(kind,twenty_per,All_Loc_num,All_Bug_num,mode): 
    result_file=open(subject+'/RESULT_'+kind+'.csv','w')
    result_record_file=open(subject+'/All-results-data/RESULT-'+kind+'-'+mode+'-'+str(exp_cursor)+'.csv','w')
    file=open(subject+"/"+kind+'.csv','r')
    lines=file.readlines()
    count=1
    total_loc=0
    total_bug=0
    
    IFA=0
    Encountered=False
    
    result_file.write('0.0,0.0\n')
    result_record_file.write('0.0,0.0\n')
    for index,each_line in enumerate(lines):
        if(total_loc<twenty_per):
            records=each_line.strip('\n').split(',')
            bug=int(records[1])
            
            if(Encountered==False):
                if(bug>0):
                    Encountered=True
                else:
                    IFA=IFA+1
            
            loc=float(records[2])
            total_loc_temp=total_loc+float(loc)
            if(total_loc_temp>twenty_per):    
                total_bug=(float(twenty_per-total_loc)/loc)*bug+total_bug
                total_loc=twenty_per
            else:
                total_bug=total_bug+int(bug)
                total_loc=total_loc_temp
            result_file.write(str(float(total_loc)/All_Loc_num)+','+str(float(total_bug)/All_Bug_num)+'\n')
            result_record_file.write(str(float(total_loc)/All_Loc_num)+','+str(float(total_bug)/All_Bug_num)+'\n')
            count=count+1
        else:
            break
    result_file.close()
    result_record_file.close()
    return IFA
##########################################################
def static_analysis():
    G = nx.DiGraph()
    findFile = open(subject+'/classgraph.dot','r')
    each_lines= findFile.readlines()
    for each_line in each_lines:
        if each_line.__contains__('>'):
            edge=each_line.split('>')
            edge[0]=edge[0][edge[0].index('\"')+1:edge[0].rindex('\"')]
            edge[1]=edge[1][edge[1].index('\"')+1:edge[1].rindex('\"')]
            if(G.has_edge(edge[0],edge[1])==False):
                G.add_edge(edge[0],edge[1])
        else:
            if each_line.count('\"')==2:
                node=each_line[each_line.index('\"')+1:each_line.rindex('\"')]
                if(G.has_node(node)==False):
                    G.add_node(node)       
    findFile.close()
    return G
##########################################################   
def label_sum(label_train):
    label_sum=0
    for each in label_train:
        label_sum=label_sum+each
    return label_sum
####################################################
def MSE(x, y):
    return np.mean((x-y)**2)
####################################################
def run_evaluation(mode):

    All_Bug_num=0
    All_Loc_num=0
    
    
    data_train,label_train=read_data_python(subject+'/train.csv')
    data_test,label_test=read_data_python(subject+'/test.csv')
    
    print(data_train.shape)
    this_column=data_train[:,3]
    print(info_gain.info_gain_ratio(label_train,this_column))
    
    package_loc_dict={}
    package_defect_dict={}
    
    defect_file=open(subject+'/Process-Origin.csv','r')
    lines=defect_file.readlines()
    for index,each_line in enumerate(lines):
        if(index!=0):
            records=each_line.strip('\n').split(',')
            class_name=records[0]
            defect_count=records[21]
            package_defect_dict[class_name]=int(defect_count)
            package_loc_dict[class_name]=records[11]

    class_test_defect_dense={}
    class_test_defect={}
    class_test_loc={}
    class_name_list=[]
    test_instance_num=0
    
    
    test_file=open(subject+'/test.csv','r')
    lines=test_file.readlines()
    for index,each_line in enumerate(lines):
        if(not index==0):
            records=each_line.strip('\n').split(',')
            class_name=records[0]
            class_name_list.append(class_name)
            class_test_defect_dense[class_name]=float(package_defect_dict[class_name])/(float(package_loc_dict[class_name])+1)
            class_test_defect[class_name]=package_defect_dict[class_name]        
            test_instance_num=test_instance_num+1
            class_test_loc[class_name]=float(package_loc_dict[class_name])
            
            All_Loc_num=All_Loc_num+float(package_loc_dict[class_name])
            All_Bug_num=All_Bug_num+int(package_defect_dict[class_name])
            
    defect_order=sorted(class_test_defect_dense.items(), key=lambda x:x[1], reverse=True)
    
    order_file=open(subject+'/optimal.csv','w')
    for each_turple in defect_order:
        each_class=each_turple[0]
        order_file.write(each_class+','+str(class_test_defect[each_class])+','+str(package_loc_dict[each_class])+'\n')
    order_file.close()
    
    reverse_order=sorted(class_test_defect_dense.items(), key=lambda x:x[1], reverse=False)
    reverse_file=open(subject+'/worst.csv','w')
    for each_turple in reverse_order:
        each_class=each_turple[0]
        reverse_file.write(each_class+','+str(class_test_defect[each_class])+','+str(package_loc_dict[each_class])+'\n')
    reverse_file.close()
    
    if(label_sum(label_train)>(len(label_train)/2)):
        print("The training data does not need balance.")
        predprob_auc,predprob,precision,recall,fmeasure,auc,mcc=classifier_output(classifier,data_train,label_train,data_test,label_test,grid_sear=True)
        print(precision,recall,fmeasure,auc,mcc)
        if(mode=="origin"):
            Precision_list_origin.append(precision)
            Recall_list_origin.append(recall)
            F_measure_list_origin.append(fmeasure)
            AUC_list_origin.append(auc)
            MCC_list_origin.append(mcc)
        elif(mode=="vector"):
            Precision_list_vector.append(precision)
            Recall_list_vector.append(recall)
            F_measure_list_vector.append(fmeasure)
            AUC_list_vector.append(auc)
            MCC_list_vector.append(mcc)
        elif(mode=="all"):
            Precision_list_all.append(precision)
            Recall_list_all.append(recall)
            F_measure_list_all.append(fmeasure)
            AUC_list_all.append(auc)
            MCC_list_all.append(mcc)
    else:
#         smo = SVMSMOTE()
#         data_bin_, label_bin_= smo.fit_sample(data_train, label_train)
        opt_para=[5,200,2]
        data_bin_, label_bin_=smote_wrapper(opt_para, data_train, label_train)
        predprob_auc,predprob,precision,recall,fmeasure,auc,mcc=classifier_output(classifier,data_bin_,label_bin_,data_test,label_test,grid_sear=True)#False is only for debugging.
        print(precision,recall,fmeasure,auc,mcc)
        if(mode=="origin"):
            Precision_list_origin.append(precision)
            Recall_list_origin.append(recall)
            F_measure_list_origin.append(fmeasure)
            AUC_list_origin.append(auc)
            MCC_list_origin.append(mcc)
        elif(mode=="vector"):
            Precision_list_vector.append(precision)
            Recall_list_vector.append(recall)
            F_measure_list_vector.append(fmeasure)
            AUC_list_vector.append(auc)
            MCC_list_vector.append(mcc)
        elif(mode=="all"):
            Precision_list_all.append(precision)
            Recall_list_all.append(recall)
            F_measure_list_all.append(fmeasure)
            AUC_list_all.append(auc)
            MCC_list_all.append(mcc)
    
    
    class_in_prediction_effortaware={}
    class_in_prediction_effortaware_coreness={}    
    for i in range(len(predprob_auc)):
        class_name=class_name_list[i]
        predict_result=predprob_auc[i]

        if(float(package_loc_dict[class_name])==0.0):
            class_in_prediction_effortaware[class_name]=0.0
        else:
            class_in_prediction_effortaware[class_name]=float(predict_result)/class_test_loc[class_name]
        
        if(float(package_loc_dict[class_name])==0.0):
            class_in_prediction_effortaware_coreness[class_name]=0.0
        else:
            if(not class_name in file_core_dict):
                class_in_prediction_effortaware_coreness[class_name]=float(predict_result)/class_test_loc[class_name]
            else:
                class_in_prediction_effortaware_coreness[class_name]=float(predict_result)*file_core_dict[class_name]/class_test_loc[class_name]
            
    model_file=open(subject+'/model.csv','w')
    effort_order=sorted(class_in_prediction_effortaware.items(), key=lambda x:x[1], reverse=True)
    for each_turple in effort_order:
        class_name=each_turple[0]
        model_file.write(class_name+','+str(class_test_defect[class_name])+','+str(package_loc_dict[class_name])+','+str(each_turple[1])+'\n')
    model_file.close()
    
    effort_file=open(subject+'/effort_core.csv','w')        
    effort_order=sorted(class_in_prediction_effortaware_coreness.items(), key=lambda x:x[1], reverse=True)
    for each_turple in effort_order:
        class_name=each_turple[0]
        effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(package_loc_dict[class_name])+','+str(each_turple[1])+'\n')
    effort_file.close()
    
    loc_order=sorted(class_test_loc.items(), key=lambda x:x[1], reverse=False)
    loc_file=open(subject+'/loc.csv','w')
    for each_turple in loc_order:
        each_class=each_turple[0]
        loc_file.write(each_class+','+str(class_test_defect[each_class])+','+str(package_loc_dict[each_class])+'\n')
    loc_file.close()
    
    
    class_in_prediction_effortaware_positive={}
    class_in_prediction_effortaware_negative={}
    
    class_in_prediction_core_positive={}
    class_in_prediction_core_negative={}
    
    for i in range(len(predprob_auc)):
        class_name=class_name_list[i]
        predict_result=predprob_auc[i]
        
        if(predict_result>=0.5):
            if(float(package_loc_dict[class_name])==0.0):
                class_in_prediction_effortaware_positive[class_name]=0.0
                class_in_prediction_core_positive[class_name]=0.0
            else:
                class_in_prediction_effortaware_positive[class_name]=float(predict_result)/class_test_loc[class_name]
                if(not class_name in file_core_dict):
                    class_in_prediction_core_positive[class_name]=float(predict_result)/class_test_loc[class_name]
                else:
                    class_in_prediction_core_positive[class_name]=float(predict_result)*file_core_dict[class_name]/class_test_loc[class_name]
        else:
            if(float(package_loc_dict[class_name])==0.0):
                class_in_prediction_effortaware_negative[class_name]=0.0
                class_in_prediction_core_negative[class_name]=0.0
            else:
                class_in_prediction_effortaware_negative[class_name]=float(predict_result)/class_test_loc[class_name]
                if(not class_name in file_core_dict):
                    class_in_prediction_core_negative[class_name]=float(predict_result)/class_test_loc[class_name]
                else:
                    class_in_prediction_core_negative[class_name]=float(predict_result)*file_core_dict[class_name]/class_test_loc[class_name]
    
    effort_file=open(subject+'/effort_CBS.csv','w')         
    effort_order=sorted(class_in_prediction_effortaware_positive.items(), key=lambda x:x[1], reverse=True)
    for each_turple in effort_order:
        class_name=each_turple[0]
        effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(package_loc_dict[class_name])+','+str(each_turple[1])+'\n')
    effort_order=sorted(class_in_prediction_effortaware_negative.items(), key=lambda x:x[1], reverse=True)
    for each_turple in effort_order:
        class_name=each_turple[0]
        effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(package_loc_dict[class_name])+','+str(each_turple[1])+'\n')
    effort_file.close()
    
    effort_file=open(subject+'/effort_CBS_core.csv','w') 
    effort_order=sorted(class_in_prediction_core_positive.items(), key=lambda x:x[1], reverse=True)
    for each_turple in effort_order:
        class_name=each_turple[0]
        effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(package_loc_dict[class_name])+','+str(each_turple[1])+'\n')
    effort_order=sorted(class_in_prediction_core_negative.items(), key=lambda x:x[1], reverse=True)
    for each_turple in effort_order:
        class_name=each_turple[0]
        effort_file.write(class_name+','+str(class_test_defect[class_name])+','+str(package_loc_dict[class_name])+','+str(each_turple[1])+'\n')
    effort_file.close()
    
    twenty_per=float(All_Loc_num)/5
 
    IFA_model=output_results('model',twenty_per,All_Loc_num,All_Bug_num,mode)      
    output_results('optimal',twenty_per,All_Loc_num,All_Bug_num,mode)
    output_results('worst',twenty_per,All_Loc_num,All_Bug_num,mode)
    output_results('loc',twenty_per,All_Loc_num,All_Bug_num,mode)
    output_results('effort_core',twenty_per,All_Loc_num,All_Bug_num,mode)
    IFA_CBS=output_results('effort_CBS',twenty_per,All_Loc_num,All_Bug_num,mode)
    output_results('effort_CBS_core',twenty_per,All_Loc_num,All_Bug_num,mode)
    
    if(mode=='origin'):
        IFA_list_origin.append(IFA_model)
        IFA_list_CBS_origin.append(IFA_CBS)
    if(mode=='vector'):
        IFA_list_vector.append(IFA_model)
        IFA_list_CBS_vector.append(IFA_CBS)
    if(mode=='all'):
        IFA_list_all.append(IFA_model)
        IFA_list_CBS_all.append(IFA_CBS)
    
    result_file=open(subject+"/ALL_POPT_Record"+"_"+mode+".csv","a")
    
    result_class_file=open(subject+"/ALL_Classification_Record"+"_"+mode+".csv","a")
    
    model_matrix = numpy.loadtxt(open(subject+"/RESULT_model.csv","rb"),delimiter=",",skiprows=0)
    optimal_matrix = numpy.loadtxt(open(subject+"/RESULT_optimal.csv","rb"),delimiter=",",skiprows=0)
    worst_matrix = numpy.loadtxt(open(subject+"/RESULT_worst.csv","rb"),delimiter=",",skiprows=0)
    loc_matrix = numpy.loadtxt(open(subject+"/RESULT_loc.csv","rb"),delimiter=",",skiprows=0)
    effortcore_matrix = numpy.loadtxt(open(subject+"/RESULT_effort_core.csv","rb"),delimiter=",",skiprows=0)
    CBS_matrix = numpy.loadtxt(open(subject+"/RESULT_effort_CBS.csv","rb"),delimiter=",",skiprows=0)
    CBScore_matrix = numpy.loadtxt(open(subject+"/RESULT_effort_CBS_core.csv","rb"),delimiter=",",skiprows=0)
    
    model=numpy.trapz(model_matrix[:,1],x=model_matrix[:,0])
    optimal=numpy.trapz(optimal_matrix[:,1],x=optimal_matrix[:,0])
    
    
    worst=numpy.trapz(worst_matrix[:,1],x=worst_matrix[:,0])
    loc=numpy.trapz(loc_matrix[:,1],x=loc_matrix[:,0])
    effortcore=numpy.trapz(effortcore_matrix[:,1],x=effortcore_matrix[:,0])
    CBS=numpy.trapz(CBS_matrix[:,1],x=CBS_matrix[:,0])
    CBScore=numpy.trapz(CBScore_matrix[:,1],x=CBScore_matrix[:,0])
    
    P_opt_model=(model-worst)/(optimal-worst)
    P_opt_loc=(loc-worst)/(optimal-worst)
    P_opt_effortcore=(effortcore-worst)/(optimal-worst)
    P_opt_CBS=(CBS-worst)/(optimal-worst)
    P_opt_CBScore=(CBScore-worst)/(optimal-worst)
    
    result_class_file.write(str(precision)+','+str(recall)+','+str(fmeasure)+','+str(auc)+','+str(mcc)+"\n")
    result_class_file.flush()
    result_class_file.close()
    
    result_file.write(str(P_opt_model)+','+str(P_opt_effortcore)+','+str(P_opt_CBS)+','+str(P_opt_CBScore)+"\n")
    result_file.flush()
    result_file.close()
    
    return P_opt_model,P_opt_effortcore,P_opt_CBS,P_opt_CBScore
####################################################
def make_prediction(binary_text,train_index,test_index,mode):
    first_line=binary_text[0]
    out_train=open(subject+'/train.csv','w')
    out_test=open(subject+'/test.csv','w')

    out_train.write(first_line)
    out_test.write(first_line)

    for each_index in train_index:
        out_train.write(binary_text[each_index+1])
    for each_index in test_index:
        out_test.write(binary_text[each_index+1])

    out_train.close()
    out_test.close()
    P_opt_model,P_opt_effortcore,P_opt_CBS,P_opt_CBScore=run_evaluation(mode)
    return P_opt_model,P_opt_effortcore,P_opt_CBS,P_opt_CBScore

####################################################
def average_value(list):
    return float(sum(list))/len(list)
####################################################

time_1=datetime.datetime.now()
date_string=time_1.strftime('%b%d')
print(date_string)
classifier="RandomForest"
embedding_model="ProNE"

##########################################################
all_F1_file=open('All-F-1-%s-%s-%s.csv'%(embedding_model,classifier,date_string),'w')
all_F1_file.write("Subject,F-1-origin,F-1-vector,F-1-all\n")
 
all_precision_file=open('All-Precision-%s-%s-%s.csv'%(embedding_model,classifier,date_string),'w')
all_precision_file.write("Subject,Precision-origin,Precision-vector,Precision-all\n")

all_MSE_file=open('All-MSE-%s-%s-%s.csv'%(embedding_model,classifier,date_string),'w')
all_MSE_file.write("Subject,MES_Result\n")
 
all_recall_file=open('All-Recall-%s-%s-%s.csv'%(embedding_model,classifier,date_string),'w')
all_recall_file.write("Subject,Recall-origin,Recall-vector,Recall-all\n")
 
all_AUC_file=open('All-AUC-%s-%s-%s.csv'%(embedding_model,classifier,date_string),'w')
all_AUC_file.write("Subject,AUC-origin,AUC-vector,AUC-all\n")
 
all_mcc_file=open('All-MCC-%s-%s-%s.csv'%(embedding_model,classifier,date_string),'w')
all_mcc_file.write("Subject,MCC-origin,MCC-vector,MCC-all\n")

all_Popt_file=open('All-Popt-%s-%s-%s.csv'%(embedding_model,classifier,date_string),'w')
all_Popt_file.write("Subject,Popt-origin,Popt-vector,Popt-all,CBS-origin,CBS-vector,CBS-all\n")

all_IFA_file=open('All-IFA-%s-%s-%s.csv'%(embedding_model,classifier,date_string),'w')
all_IFA_file.write("Subject,IFA-origin,IFA-vector,IFA-all,IFA-origin,IFA-vector,IFA-all\n")

all_IGR_file=open('All-IGR-%s-%s-%s.csv'%(embedding_model,classifier,date_string),'w')


Statistical_Popt=open('All-STATISTICAL-Popt-%s-%s-%s.csv'%(embedding_model,classifier,date_string),"w")
Statistical_Popt.write("Subject,origin vs vector,origin vs all,CBS vs CBS-vector,CBS vs CBS-all\n")

Statistical_IFA=open('All-STATISTICAL-IFA-%s-%s-%s.csv'%(embedding_model,classifier,date_string),"w")
Statistical_IFA.write("Subject,origin vs vector,origin vs all,CBS vs CBS-vector,CBS vs CBS-all\n")

Statistical_MCC=open('All-STATISTICAL-MCC-%s-%s-%s.csv'%(embedding_model,classifier,date_string),"w")
Statistical_MCC.write("Subject,origin vs vector,origin vs all\n")

Statistical_AUC=open('All-STATISTICAL-AUC-%s-%s-%s.csv'%(embedding_model,classifier,date_string),"w")
Statistical_AUC.write("Subject,origin vs vector,origin vs all\n")

Statistical_F1=open('All-STATISTICAL-F-1-%s-%s-%s.csv'%(embedding_model,classifier,date_string),"w")
Statistical_F1.write("Subject,origin vs vector,origin vs all\n")

dict_file=open('Subject.dict','r')
lines=dict_file.readlines()
for each_line in lines:
    records=each_line.strip().split(',')
    subject=records[0]
    bug_file=records[1]
    
    
    G=static_analysis()
    G.remove_edges_from(nx.selfloop_edges(G))
    file_core_dict={}
    j=1
    while(j<100):
        defect_num=0
        count=0
        total_loc=0
        G1=nx.k_core(G,j)
        node_list=G1.nodes()
        if(len(node_list)==0):
            break
        for node in node_list:
            file_core_dict[node]=j
        j=j+1
    
    class_name_dict={}
    class_name_file=open(subject+'/node_mapping','r')
    lines=class_name_file.readlines()
    for index,each_line in enumerate(lines):
        records=each_line.strip('\n').split(',')
        class_name_dict[records[1]]=records[0]
    
    class_vector_dict={}
    vector_file=open(subject+'/classgraph.emd','r')
    lines=vector_file.readlines()
    for index,each_line in enumerate(lines):
        if(not index==0):
            class_index=each_line[:each_line.index(' ')]
            vector=each_line[each_line.index(' ')+1:].strip('\n')
            class_vector_dict[class_name_dict[class_index]]=vector
            
    
    core_dict=nx.core_number(G)
    core_value=[]
    embeddings=[]
    for each_class in class_vector_dict:
        if(each_class in core_dict):
            embedding=np.array(list(map(float,class_vector_dict[each_class].split(' '))))
            embeddings.append(embedding)
            core_value.append(core_dict[each_class])
    
    core_value_array=np.array(core_value)
    embeddings_array=np.array(embeddings)
    print(core_value_array.shape)
    print(embeddings_array.shape)
    lr = LinearRegression(n_jobs=-1)
    y_pred = cross_val_predict(lr, embeddings_array, core_value_array)
    print(y_pred.shape)
    result=MSE(y_pred, core_value_array)/np.mean(core_value_array)
    print(result)
    
    all_MSE_file.write(subject+','+str(result)+'\n')
    all_MSE_file.flush()
    
    
    
    out_binary=open(subject+'/Process-Binary.csv','w')
    out_origin=open(subject+'/Process-Origin.csv','w')
    out_vector=open(subject+'/Process-Vector.csv','w')
    out_all=open(subject+'/Process-All.csv','w')
    
    out_binary.write("name,wmc,dit,noc,cbo,rfc,lcom,ca,ce,npm,lcom3,loc,dam,moa,mfa,cam,ic,cbm,amc,max_cc,avg_cc,bug\n")
    out_origin.write("name,wmc,dit,noc,cbo,rfc,lcom,ca,ce,npm,lcom3,loc,dam,moa,mfa,cam,ic,cbm,amc,max_cc,avg_cc,bug\n")
    out_all.write("name,wmc,dit,noc,cbo,rfc,lcom,ca,ce,npm,lcom3,loc,dam,moa,mfa,cam,ic,cbm,amc,max_cc,avg_cc")
    
    vector_index=0
    out_vector.write('name')
    while(vector_index<32):
        out_vector.write(',vector_'+str(vector_index))
        out_all.write(',vector_'+str(vector_index))
        vector_index=vector_index+1
    
    out_vector.write(',bug\n')
    out_all.write(',bug\n')
    
    bug_record_file=open(subject+'/'+bug_file,'r')
    class_record=open(subject+'/ClassNotInCDN.csv','w')
    
    lines=bug_record_file.readlines()
    for index,each_line in enumerate(lines):
        if(not index==0):
            records=each_line.strip('\n').split(',')
            class_name=records[2]
            original_metrics=each_line[each_line.index(class_name)+len(class_name)+1:each_line.rindex(',')]
            defect_count=int(each_line[each_line.rindex(',')+1:].strip('\n'))
            if(defect_count>0):
                if(class_name in class_vector_dict):
                    out_origin.write(class_name+','+original_metrics+','+str(defect_count)+'\n')
                    out_binary.write(class_name+','+original_metrics+',1\n')
                    out_vector.write(class_name+','+class_vector_dict[class_name].replace(' ',',')+',1\n')
                    out_all.write(class_name+','+original_metrics+','+class_vector_dict[class_name].replace(' ',',')+',1\n')
                else:
                    class_record.write(class_name+','+original_metrics+'\n')
            else:
                if(class_name in class_vector_dict):
                    out_origin.write(class_name+','+original_metrics+',0\n')
                    out_binary.write(class_name+','+original_metrics+',0\n')
                    out_vector.write(class_name+','+class_vector_dict[class_name].replace(' ',',')+',0\n')
                    out_all.write(class_name+','+original_metrics+','+class_vector_dict[class_name].replace(' ',',')+',0\n')
                else:
                    class_record.write(class_name+','+original_metrics+'\n')
            
    out_origin.close()
    out_binary.close()
    out_vector.close()
    out_all.close()
    class_record.close()
    
    if(os.path.exists(subject+'/All-results-data')):
        shutil.rmtree(subject+'/All-results-data')
    os.mkdir(subject+'/All-results-data')
    if(os.path.exists(subject+'/ALL_POPT_Record_origin.csv')):
        os.remove(subject+'/ALL_POPT_Record_origin.csv')
    if(os.path.exists(subject+'/ALL_POPT_Record_vector.csv')):
        os.remove(subject+'/ALL_POPT_Record_vector.csv')
    if(os.path.exists(subject+'/ALL_POPT_Record_all.csv')):
        os.remove(subject+'/ALL_POPT_Record_all.csv')
    if(os.path.exists(subject+'/ALL_Classification_Record_origin.csv')):
        os.remove(subject+'/ALL_Classification_Record_origin.csv')
    if(os.path.exists(subject+'/ALL_Classification_Record_vector.csv')):
        os.remove(subject+'/ALL_Classification_Record_vector.csv')
    if(os.path.exists(subject+'/ALL_Classification_Record_all.csv')):
        os.remove(subject+'/ALL_Classification_Record_all.csv')
    
    origin_text=read_text_data(subject+'/Process-Origin.csv')
    binary_text=read_text_data(subject+'/Process-Binary.csv')
    origin_data=read_data(subject+'/Process-Binary.csv')
    
    vector_text=read_text_data(subject+'/Process-Vector.csv')
    
    all_text=read_text_data(subject+'/Process-All.csv')
    
    data_all,label_all=read_data_python(subject+'/Process-All.csv')
    print(data_all.shape)
    IGR_result_file=open(subject+'/IGR_results.csv','w')
    num_metrics=data_all.shape[1]
    metrics_cursor=0
    IGR_dict={}
    
    while(metrics_cursor<num_metrics):
        this_column=data_all[:,metrics_cursor]
        IGR=info_gain.info_gain_ratio(label_all,this_column)
        IGR_dict[metrics_cursor]=IGR
        metrics_cursor=metrics_cursor+1
    
    all_IGR_file.write(subject)    
    IGR_order=sorted(IGR_dict.items(), key=lambda x:x[1], reverse=True)
    IGR_counter=0
    for each_turple in IGR_order:
        IGR_result_file.write(str(each_turple[0])+','+str(each_turple[1])+'\n')
        IGR_result_file.flush()
        if(IGR_counter<10):
            all_IGR_file.write(','+str(each_turple[0]))
            IGR_counter=IGR_counter+1
    all_IGR_file.write('\n')
    all_IGR_file.flush()
    
    
    P_opt_list_origin=[]
    P_opt_list_vector=[]
    P_opt_list_all=[]
    
    IFA_list_origin=[]
    IFA_list_vector=[]
    IFA_list_all=[]
    
    P_opt_list_origin_core=[]
    P_opt_list_vector_core=[]
    P_opt_list_all_core=[]
    
    P_opt_list_CBS_origin=[]
    P_opt_list_CBS_vector=[]
    P_opt_list_CBS_all=[]
    
    IFA_list_CBS_origin=[]
    IFA_list_CBS_vector=[]
    IFA_list_CBS_all=[]
    
    P_opt_list_CBS_origin_core=[]
    P_opt_list_CBS_vector_core=[]
    P_opt_list_CBS_all_core=[]
    
    F_measure_list_origin=[]
    F_measure_list_vector=[]
    F_measure_list_all=[]
    
    AUC_list_origin=[]
    AUC_list_vector=[]
    AUC_list_all=[]
    
    MCC_list_origin=[]
    MCC_list_vector=[]
    MCC_list_all=[]
    
    Precision_list_origin=[]
    Precision_list_vector=[]
    Precision_list_all=[]
    
    Recall_list_origin=[]
    Recall_list_vector=[]
    Recall_list_all=[]
    
    
    exp_cursor=1
    kf = RepeatedKFold(n_splits=5, n_repeats=5)#We can modify n_repeats when debugging.
    for train_index, test_index in kf.split(origin_data):
        P_opt_model,P_opt_effortcore,P_opt_CBS,P_opt_CBScore=make_prediction(binary_text, train_index, test_index,"origin")
        P_opt_list_origin.append(P_opt_model)
        P_opt_list_origin_core.append(P_opt_effortcore)
        P_opt_list_CBS_origin.append(P_opt_CBS)
        P_opt_list_CBS_origin_core.append(P_opt_CBScore)
        
        
        P_opt_model,P_opt_effortcore,P_opt_CBS,P_opt_CBScore=make_prediction(vector_text, train_index, test_index,"vector")
        P_opt_list_vector.append(P_opt_model)
        P_opt_list_vector_core.append(P_opt_effortcore)
        P_opt_list_CBS_vector.append(P_opt_CBS)
        P_opt_list_CBS_vector_core.append(P_opt_CBScore)
        
        P_opt_model,P_opt_effortcore,P_opt_CBS,P_opt_CBScore=make_prediction(all_text, train_index, test_index,"all")
        P_opt_list_all.append(P_opt_model)
        P_opt_list_all_core.append(P_opt_effortcore)
        P_opt_list_CBS_all.append(P_opt_CBS)
        P_opt_list_CBS_all_core.append(P_opt_CBScore)
        
        
        exp_cursor=exp_cursor+1
    
    print(Recall_list_origin)
    print(Recall_list_vector)
    print(Recall_list_all)
    
    all_F1_file.write(subject+','+str(average_value(F_measure_list_origin))+','+str(average_value(F_measure_list_vector))+','+str(average_value(F_measure_list_all))+'\n')
    all_F1_file.flush()
    all_precision_file.write(subject+','+str(average_value(Precision_list_origin))+','+str(average_value(Precision_list_vector))+','+str(average_value(Precision_list_all))+'\n')
    all_precision_file.flush()
    all_recall_file.write(subject+','+str(average_value(Recall_list_origin))+','+str(average_value(Recall_list_vector))+','+str(average_value(Recall_list_all))+'\n')
    all_recall_file.flush()
    all_AUC_file.write(subject+','+str(average_value(AUC_list_origin))+','+str(average_value(AUC_list_vector))+','+str(average_value(AUC_list_all))+'\n')
    all_AUC_file.flush()
    all_mcc_file.write(subject+','+str(average_value(MCC_list_origin))+','+str(average_value(MCC_list_vector))+','+str(average_value(MCC_list_all))+'\n')
    all_mcc_file.flush()
    
    
    all_Popt_file.write(subject+','+str(average_value(P_opt_list_origin))+','+str(average_value(P_opt_list_vector))+','+str(average_value(P_opt_list_all))+','+str(average_value(P_opt_list_CBS_origin))+','+str(average_value(P_opt_list_CBS_vector))+','+str(average_value(P_opt_list_CBS_all))+'\n')
    all_Popt_file.flush()
    
    all_IFA_file.write(subject+','+str(average_value(IFA_list_origin))+','+str(average_value(IFA_list_vector))+','+str(average_value(IFA_list_all))+','+str(average_value(IFA_list_CBS_origin))+','+str(average_value(IFA_list_CBS_vector))+','+str(average_value(IFA_list_CBS_all))+'\n')
    all_IFA_file.flush()
    
    Statistical_Popt.write(subject+','+str(stats.wilcoxon(P_opt_list_origin,P_opt_list_vector,alternative='less'))+','+str(stats.wilcoxon(P_opt_list_origin,P_opt_list_all,alternative='less'))+','+str(stats.wilcoxon(P_opt_list_CBS_origin,P_opt_list_CBS_vector,alternative='less'))+','+str(stats.wilcoxon(P_opt_list_CBS_origin,P_opt_list_CBS_all,alternative='less'))+'\n')
    Statistical_Popt.flush()
    
    Statistical_IFA.write(subject+','+str(stats.wilcoxon(IFA_list_origin,IFA_list_vector,alternative='less'))+','+str(stats.wilcoxon(IFA_list_origin,IFA_list_all,alternative='less'))+','+str(stats.wilcoxon(IFA_list_CBS_origin,IFA_list_CBS_vector,alternative='less'))+','+str(stats.wilcoxon(IFA_list_CBS_origin,IFA_list_CBS_all,alternative='less'))+'\n')
    Statistical_IFA.flush()
    
    Statistical_MCC.write(subject+','+str(stats.wilcoxon(MCC_list_origin,MCC_list_vector,alternative='less'))+','+str(stats.wilcoxon(MCC_list_origin,MCC_list_all,alternative='less'))+'\n')
    Statistical_MCC.flush()
    Statistical_AUC.write(subject+','+str(stats.wilcoxon(AUC_list_origin,AUC_list_vector,alternative='less'))+','+str(stats.wilcoxon(AUC_list_origin,AUC_list_all,alternative='less'))+'\n')
    Statistical_AUC.flush()
    Statistical_F1.write(subject+','+str(stats.wilcoxon(F_measure_list_origin,F_measure_list_vector,alternative='less'))+','+str(stats.wilcoxon(F_measure_list_origin,F_measure_list_all,alternative='less'))+'\n')
    Statistical_F1.flush()
