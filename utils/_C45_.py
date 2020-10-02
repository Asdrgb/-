# -*- coding: utf-8 -*-

import numpy as np
from utils._id3_ import calculateGain,calculateEntropy,Node,calculateMostClass









def calculateGainRatio(data,entD,cla):
    n = data.shape[0]
    
    gain = calculateGain(data,entD,cla)
    
    a_val_list = np.unique(data[:,0])
    
    IVa = 0
    
    for i,val in enumerate(a_val_list):
        m = len(data[data==val])
        p = m/n
        IVa += p*np.log2(p) 
    
    gainratio_ = gain/IVa
    
    return -gainratio_





def selectBestAtrributetoDivide_GainRatio(data,cla,att):
    m = data.shape[1]
    allresult = []
    atrr_index = 0
    Biggest_GainRatio = 0
    entD = calculateEntropy(data[1:,-1],cla)
    for i,val in enumerate(att):
        its = list(data[0,:]).index(val[0])
        gainRatio = calculateGainRatio(data[1:,[its,m-1]],entD,cla)
        allresult.append((data[0][its],gainRatio))
        if Biggest_GainRatio < gainRatio:
            Biggest_GainRatio = gainRatio
            atrr_index = i
    
    return atrr_index,allresult








def create_C45(data,cla,a_val,depth):  #cla=['是','否'] a_val=[['色泽'，‘青绿’，‘乌黑’,..],.....,['触感'，‘硬滑’，...],['好瓜'，‘是’，‘否’]]
    node = Node()#创建根节点
    node.alt = a_val[:]  #浅拷贝
    node.depth = depth
    if Node.maxdepth < node.depth:  #更新最大层
        Node.maxdepth = node.depth
    #print(a_val)
    #返回条件1
    if len(np.unique(data[1:,-1]))==1:
        c = np.unique(data[1:,-1])[0]
        node.val = c
        Node.numsleaf += 1
        node.child = None  #此节点为叶节点
        return node
    
    #
    #返回条件2
    #
    #print(len(a_val))
    if len(a_val)==0:
        node.val = calculateMostClass(data[:,-1])
        Node.numsleaf += 1
        
        node.child = None   #此节点为叶节点
        return node
    
    
    a_index,allresult = selectBestAtrributetoDivide_GainRatio(data,cla,a_val)#计算最优划分属性所在的列索引
    node.midresult = allresult
    best_att = a_val[a_index] #取出最优属性及其属性值 best_att=[属性,属性值1，...，属性值n]
    
    node.val = best_att[0] #设定节点值
    
    a_val.pop(a_index) #移除最优划分属性，为下一次划分做准备
    
    first_row = data[0,:] #提取第一行，所有子集都要和第一行组合为一个新的子集
    
    its = list(first_row).index(best_att[0])
    
    for i,val in enumerate(best_att[1:]):   # best_att[1:] = [‘属性值1’，‘属性值2’，。。。‘属性值n’]
        subdata = data[data[:,its]==val]  #提取数据
        
        #返回条件3,该属性值所对应的样本集合为空时，该属性值所对应的节点为叶子节点
        if len(subdata) == 0:
            child_node = Node()
            label = calculateMostClass(data[:,-1])
            child_node.val = label
            Node.numsleaf += 1
            if child_node.maxdepth < depth+1:child_node.maxdepth = depth+1
            child_node.child = None #此节点为叶节点
            node.child.append((val,child_node))
            continue
            
        subdata = np.vstack([first_row,subdata])
        child_node = create_C45(subdata,cla,a_val[:],depth+1) #a_val[:]的作用是浅拷贝，因为直接传递的a_val是引用
        node.child.append((val,child_node))                 #引用相当于指针，每次对引用的修改都是对原始数据的修改
    
    
    return node




