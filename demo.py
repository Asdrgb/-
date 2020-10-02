# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:52:36 2020

@author: win、
"""

from models.ID3 import ID3
from models.C45 import C45
from utils.parse import parse_data,tree2dict
from utils.plot import createPlot

data_path = 'data/xigua21_train.txt'
data,cla,att = parse_data(data_path)


tree = ID3(data,cla,att)
d_tree = tree2dict(tree)
createPlot(d_tree,'id3')

#tree2 = C45(data,cla,att)
#d_tree2 = tree2dict(tree2)
#createPlot(d_tree2,'c4.5')




