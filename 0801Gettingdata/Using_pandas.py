#!/usr/bin/python
# -*- coding:euc-kr -*-

import pandas as pd
import os
os.chdir('C:\\Studying\\GrowthHackers\\0801Gettingdata')

# data= pd.read_csv(r'jeju_beach_2017.csv', encoding= "euc-kr", index_col= ['����'])
data= pd.read_table(r'jeju_beach_2017.csv', encoding= "euc-kr", sep=",", index_col= ['������', '�غ���'])
data= data.drop('����', axis= 1)

# print(data)
print(data.ix['���ֽ�', ['ȭ���(��)']])
