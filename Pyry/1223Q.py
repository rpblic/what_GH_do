import numpy as np
import pandas as pd
import os
os.chdir('C:\\Studying\\myvenv\\GrowthHackers\\Pyry')

data_as_list= [\
['Adele', '4:46', 'Skyfall', 2012],\
['Eminem', '5:26', 'Lose Yourself', 2002],\
['Pink Floyd', '3:59', 'Another Brick in the Wall', 1979],\
['Kings of Leon', '3:51', 'Use Somebody', 2008],\
['Bruno Mars', '2:58', 'Treasure', 2013],]
label_as_list= ['Artist', 'Length', 'Title', 'Year']
data= pd.DataFrame(data_as_list, columns= label_as_list)
print(data)
#           Artist Length                      Title  Year
# 0          Adele   4:46                    Skyfall  2012
# 1         Eminem   5:26              Lose Yourself  2002
# 2     Pink Floyd   3:59  Another Brick in the Wall  1979
# 3  Kings of Leon   3:51               Use Somebody  2008
# 4     Bruno Mars   2:58                   Treasure  2013
print(data[['Title', 'Artist', 'Length', 'Year']])
#                        Title         Artist Length  Year
# 0                    Skyfall          Adele   4:46  2012
# 1              Lose Yourself         Eminem   5:26  2002
# 2  Another Brick in the Wall     Pink Floyd   3:59  1979
# 3               Use Somebody  Kings of Leon   3:51  2008
# 4                   Treasure     Bruno Mars   2:58  2013

data_title= data['Title'].tolist()
data_title= [x.lower() for x in data_title]
print(data_title)
# ['skyfall', 'lose yourself', 'another brick in the wall', 'use somebody', 'treasure']
data_Q10= pd.read_excel('.\\[Module 2] Numpy & Pandas\\animals.xlsx')
data_Q10= data_Q10.loc[:, :'feathers']
print(data_Q10.head(10))
#        name  hair  feathers
# 0  aardvark     1         0
# 1  antelope     1         0
# 2      bass     0         0
# 3      bear     1         0
# 4      boar     1         0
# 5   buffalo     1         0
# 6      calf     1         0
# 7      carp     0         0
# 8   catfish     0         0
# 9      cavy     1         0
data_Q10.to_excel('.\\Q11.xlsx')
