#!/usr/bin/python
# -*- coding:euc-kr -*-

import pandas as pd
import os
os.chdir('C:\\Studying\\GrowthHackers\\0801Gettingdata')

# data= pd.read_csv(r'jeju_beach_2017.csv', encoding= "euc-kr", index_col= ['연번'])
data= pd.read_table(r'jeju_beach_2017.csv', encoding= "euc-kr", sep=",", index_col= ['행정시', '해변명'])
data= data.drop('연번', axis= 1)

# print(data)
print(data.ix['제주시', ['화장실(동)']])
