#import os
import numpy as np



#计算某个属性值的熵，data是一个数组，data=[n*类别标签列表]，n为样本数        
def calculateAtrrValEntropy(data,cla):  
    n = len(data)
    entropy = 0
    for i,val in enumerate(cla):
        
        m = len(data[data==val]) #计算类别为val的样本数量m
        
        p = m/n                 #计算m与总总样本n的比例（概率）
        if p==0:
            pp = 0
        else:
            pp = p*np.log2(p)  #保留3为小数
        entropy = entropy + pp   #累加不同类
        
    return -entropy            #返回信息熵
    
    
#计算某个属性的熵 data=[属性值,类标签] a=(属性名，索引)
def calculateAtrrEntropy(data,cla):  
    n,m = data.shape
    attr_val = np.unique(data[:,0])
    #attr_val_nums = len(attr_val)
    w_ent=0
    for i,a_val in enumerate(attr_val):
        a = data[data[:,0]==a_val][:,1]  #筛选出属性值等于 a_val 的样本，并只返回相应的类标签
        a_n = len(a)                     #属性值等于a_val的样本数量
        ent = calculateAtrrValEntropy(a,cla)  #计算属性值a_val的熵
        w_ent = w_ent + (a_n/n) * ent     #属性值熵加权累加
        
    return w_ent  #返回各属性值熵的加权累加和



#计算待划分数据集的信息熵 
def calculateEntropy(data,cla):  
    n = len(data)
    ent = 0
    for i,c in enumerate(cla):
        m = len(data[data==c])
        p = m/n
        if p==0:
            pp = 0
        else:
            pp = p*np.log2(p)
        ent = ent + pp
        
    return -ent 
        

#计算某个属性的信息增益
def calculateGain(data,entD,cla):  
    
    w_entA = calculateAtrrEntropy(data,cla)   #计算各属性值熵的加权累加和
    
    gain = entD - w_entA                 #计算该属性的信息增益
    
    return np.around(gain,3)           #返回信息增益
    






#选择最优划分属性
def selectBestAtrributetoDivide_Gain(data,cla,att):
    c_n = cla #np.unique(data[:,-1])               #类别个数
    m =  data.shape[1]
    allresult = []
    atrr_index = 0
    Biggest_Gain = 0
    entD = calculateEntropy(data[1:,-1],c_n)  #计算待划分数据的信息熵
    for i,val in enumerate(att):   #val = [属性，属性值1，属性值2，。。。，属性值n]
        its = list(data[0,:]).index(val[0])   #该属性的索引
        gain = calculateGain(data[1:,[its,m-1]],entD,c_n)  #计算该属性的信息增益
        allresult.append((data[0][its],gain))
        if Biggest_Gain < gain:                        
            Biggest_Gain = gain                         #保留最大值
            atrr_index = i                              #记住当前最大信息增益所属属性的索引(att中的索引)
            
        
    
    return atrr_index,allresult   #返回最优划分属性所在的列索引
    


        
#节点类
class Node():
    maxdepth = 0 #决策树的深度
    numsleaf = 0 #决策树的叶子节点数
    def __init__(self):
        self.val = None  #节点值
        self.child = list()  #子节点集
        self.midresult = None  #当前节点信息增益计算结果
        self.alt = None       #当前节点可选属性
        self.depth = 0        #当前节点所在层




#计算样本集中样本最多的类
def calculateMostClass(data):  #计算样本张最多的类 data = [类1，类1，类2，。。。。，类2，类1],data是数据集的最后一列
    cl = np.unique(data)
    maxnums = 0
    maxcl = None
    for i in range(len(cl)):
        nums = list(data).index(cl[i])
        if nums > maxnums:
            maxcl = cl[i]
    
    return maxcl


    
    
    
#构建决策树 
def create_ID3(data,cla,a_val,depth):  #cla=['是','否'] a_val=[['色泽'，‘青绿’，‘乌黑’,..],.....,['触感'，‘硬滑’，...],['好瓜'，‘是’，‘否’]]
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
    
    
    a_index,allresult = selectBestAtrributetoDivide_Gain(data,cla,a_val)#计算最优划分属性所在的列索引
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
        child_node = create_ID3(subdata,cla,a_val[:],depth+1) #a_val[:]的作用是浅拷贝，因为直接传递的a_val是引用
        node.child.append((val,child_node))                 #引用相当于指针，每次对引用的修改都是对原始数据的修改
    
    
    return node
    
    

        
            



    
        
        
        
         
    
          
        


               
                    
            
    
    
        
    


















