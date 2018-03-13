import os
os.getcwd()
getdat= open("Samsung_Stock_Table.csv", "rt", encoding= "UTF-8")
lines= getdat.readlines()
for elmt in lines: print(elmt, type(elmt))
keylst= lines[0].split(','); print(keylst)
keylst[-1]= keylst[-1][:-1]; print(keylst)
for elmt in lines:
    if elmt[:8]== '17.06.20':
        vallst= elmt.split(',')
        vallst[-1]= vallst[-1][:-1]
        dict_0620= {keylst[i]: vallst[i] for i in range(len(keylst))}
        print(dict_0620)
# dict_0620= [elmt.split(',') for elmt in lines if elmt[:8]=='17.06.20']
# print(dict_0620)
# print(dict_0620[-1])
# dict_0620= dict_0620[0]
# dict_0620[-1]= dict_0620[-1][:-1]; print(dict_0620)
