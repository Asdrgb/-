# -*- coding: utf-8 -*-




import numpy as np

def parse_data(path): #读入数据，解析数据，一行一个样本，每列一个属性，最后一列是标签
    
    file = open(path,encoding='utf8')
    
    data = np.loadtxt(file,dtype=np.str)
    cla = np.unique(data[1:,-1])
    a_val = []
    for i,val in enumerate(data[0,0:len(data[0,:])-1]):
        a_v =  [val]+list(np.unique(data[1:,i]))
        
        a_val.append(a_v)
    
    return data,cla,a_val



#树结构到字典的映射
def tree2dict(node):
    
    root = node 
    if root.child==None: #如果当前节点无子节点，则返回节点值
        return str(root.val)
    

    else:   #当前节点存在子节点，则获取子节点信息
        result = {}
        result[str(root.val)] = {}
        for i in range(len(root.child)):
            result[str(root.val)][root.child[i][0]] = tree2dict(root.child[i][1])#递归获取子节点信息
            
        return result   #返回一个字典，该字典中包含子节点信息