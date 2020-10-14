# -*- coding:utf-8 -*-
'''
Created on 2018年9月27日

@author: Yu Qu
'''
import networkx as nx

##########################################################
def static_analysis(subject,file_network_file,prefix):
    G = nx.DiGraph()
    findFile = open(subject+'/'+file_network_file,'r')
    each_lines= findFile.readlines()
    for each_line in each_lines:
        if each_line.__contains__('>'):
            edge=each_line.split('>');
            if(prefix in edge[0]):
                edge[0]=edge[0][edge[0].index(prefix)+len(prefix):edge[0].rindex('\"')].replace("\\","/")
            if(prefix in edge[1]):
                edge[1]=edge[1][edge[1].index(prefix)+len(prefix):edge[1].rindex('\"')].replace("\\","/") 
            if(G.has_edge(edge[0],edge[1])==False):
                G.add_edge(edge[0],edge[1])       
    findFile.close()
    return G