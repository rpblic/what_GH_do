import csv
import numpy as np
import os
os.chdir('C:\\Studying\\GrowthHackers\\0905 Network')

def data_making(filename):
    with open(filename, 'rt', newline= '') as f:
        data= csv.reader(f, delimiter= ',')
        #open csv file. 'data' is iterating type, but not a list.
    #     next(data, None)    #Deleting Index
    #     node= []
    #     q1_edge= []
    #     q2_edge= []
    #     for row in data:
    #         answer_list= [val.split(', ') for val in row[1:]]
    #         node.append(row[0])
    #         q1_edge.extend([(row[0], i) for i in answer_list[0]])
    #         q2_edge.extend([(row[0], i) for i in answer_list[1]])
    # return (node, q1_edge, q2_edge)

        for row in data:
            print(row)
            
def top_n(inp_dict, n_int):
    return sorted(inp_dict.items(), key= lambda x: x[1], reverse= True)[:n_int]

if __name__== '__main__':
    print(data_making('.\\최종 전처리 데이터.csv')[1])
