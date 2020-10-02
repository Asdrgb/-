# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决绘图中中文字符乱码的情况



decisionNode=dict(boxstyle="round",fc="0.8")
# 定义决策树的叶子结点的描述属性
leafNode=dict(boxstyle="circle",fc="0.8")

# 定义决策树的箭头属性
arrow_args=dict(arrowstyle="<-")

def retrieveTree(i):
    listOfTree=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
        {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTree[i]

#使用文本注解绘制树节点
# nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    # annotate是关于一个数据点的文本
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

## 创建绘图
#def createPlot():
#    # 类似于Matlab的figure，定义一个画布(暂且这么称呼吧)，背景为白色
#    fig=plt.figure(1,facecolor='white')
#    # 把画布清空
#    fig.clf()
#    # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图
#    # frameon表示是否绘制坐标轴矩形
#    createPlot.ax1=plt.subplot(111,frameon=False)
#    # 绘制结点
#    plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
#    plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
#    #显示画图
#    plt.show()

#createPlot()
    

# 绘制中间文本
def plotMidText(cntrPt,parentPt,txtString):
    # 求中间点的横坐标
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    # 求中间点的纵坐标
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    # 绘制树结点
    createPlot.ax1.text(xMid,yMid,txtString)

# 绘制决策树
def plotTree(myTree,parentPt,nodeTxt):  #(tree,(0.5 , 1.0),nodeset)
    # 定义并获得决策树的叶子结点数
    numLeafs=getNumsLeaf(myTree)
    #depth=getTreeDepth(myTree)
    # 得到第一个特征
    firstStr=list(myTree.keys())[0]
    # 计算坐标，x坐标为当前树的叶子结点数目除以整个树的叶子结点数再除以2，y为起点
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    # 绘制中间结点，即决策树结点，也是当前树的根结点
    plotMidText(cntrPt,parentPt,nodeTxt)
    # 绘制决策树结点
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    # 根据firstStr找到对应的值
    secondDict=myTree[firstStr]
    # 因为进入了下一层，所以y的坐标要变 ，图像坐标是从左上角为原点
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    # 遍历secondDict
    for key in secondDict.keys():
        # 如果secondDict[key]为一棵子决策树，即字典
        if type(secondDict[key]).__name__=='dict':
            # 递归的绘制决策树
            plotTree(secondDict[key],cntrPt,str(key))
        # 若secondDict[key]为叶子结点
        else:
            # 计算叶子结点的横坐标
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            # 绘制叶子结点
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            #特征值
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    # 计算纵坐标
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD




def getNumsLeaf(mytree):
    numleaf = 0
    
    firstnode = list(mytree.keys())  # 分支
    #print(firstnode)
    #seconddict = mytree[firstnode]   #
    #print(seconddict)
    for key in firstnode:  # 
        if type(mytree[key]).__name__ == 'dict':
            numleaf += getNumsLeaf(mytree[key])   # 
        else:
            numleaf += 1
    
    return numleaf


def getTreeDepth(myTree):
    # 定义树的深度
    maxDepth=0
    # 获得myTree的第一个键值，即第一个特征，分割的标签
    firstStr=list(myTree.keys())
    # 根据键值得到对应的值，即根据第一个特征分类的结果
    #print(firstStr)
    for key in firstStr:
        # 如果myTree[key]为一个字典
        if type(myTree[key]).__name__=='dict':
            # 则当前树的深度等于1加上secondDict的深度，只有当前点为决策树点深度才会加1
            thisDepth=1+getTreeDepth(myTree[key])
        # 如果secondDict[key]为叶子结点
        else:
            # 则将当前树的深度设为1
            thisDepth=1
        # 比较当前树的深度与最大数的深度
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    # 返回树的深度
    return maxDepth



   
##主函数 绘图
def createPlot(inTree,name=None):
    # 定义一块画布
    fig=plt.figure(name,facecolor='white')
    
    # 清空画布
    fig.clf()
    
    # 定义横纵坐标轴，无内容
    axprops=dict(xticks=[],yticks=[])
    # 绘制图像，无边框，无坐标轴
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    # plotTree.totalW保存的是树的宽
    plotTree.totalW=float(getNumsLeaf(inTree))
    # plotTree.totalD保存的是树的高
    plotTree.totalD=float(getTreeDepth(inTree))
    # 决策树起始横坐标
    plotTree.xOff=-0.5/plotTree.totalW
    # 决策树的起始纵坐标
    plotTree.yOff=1.0
    # 绘制决策树
    plotTree(inTree,(0.5,1.0),'')
    
    # 显示图像
    plt.show()
    












