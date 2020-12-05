# -*- coding:utf-8 -*-
'''
Created on 2018.3.18

@author: Yu Qu
'''
import networkx as nx
from math import *

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
conf_file=open('Subject.dict')
lines=conf_file.readlines()
for each_line in lines:
    records=each_line.strip().split(',')
    subject=records[0]
    bug_file=records[1]
    edge_file=open(subject+'/edgelist','w')
    mapping_file=open(subject+'/node_mapping','w')
    
    G=static_analysis()
    print('Number of Nodes:'+str(G.number_of_nodes()))
    print('Number of Edges:'+str(G.size()))    
    nx.write_gexf(G,subject+"/GraphForTraditionalNetworkMetrics.gexf")
    
    node_list=list(G.nodes())
    for e in G.edges():
        source=e[0]
        dist=e[1]
        s=node_list.index(source)
        d=node_list.index(dist)
        edge_file.write(str(s)+' '+str(d)+'\n')
    index=0
    for each_node in node_list:
        mapping_file.write(each_node+','+str(index)+'\n')
        index=index+1
    
