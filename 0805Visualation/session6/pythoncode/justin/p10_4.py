"""10.4 척도조절"""

""" 0. 사전작업 """

# 미리 필요한 함수 정의하기

from linear_algebra import shape, get_row, get_column, make_matrix, \
    vector_mean, vector_sum, dot, magnitude, vector_subtract, scalar_multiply
from stats import standard_deviation, mean
import math

def shape(A): # 행 갯수 열 갯수 반환 / np.shape()
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # number of elements in first row
    return num_rows, num_cols

def make_matrix(num_rows, num_cols, entry_fn): # 행렬 만드는 함수
    """returns a num_rows x num_cols matrix
    whose (i,j)th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) # given i, create a list
            for j in range(num_cols)] # [entry_fn(i, 0), ... ]
            for i in range(num_rows)] # create one list for each i

# 파일 읽어오기

whole = []
f = open("HR_comma_sep2.csv","r",encoding = "utf-8")
lines = f.readlines()

for line in lines :
    new_element = line.split(',')[0:5] # 우리가 분석할 부분(수치로 나와있는 부분) 추출
    whole.append(new_element)

data = []
for contents in whole[1:] :
    contents_int = [float(i) for i in contents]
    data.append(contents_int)

print(data)


""" 1. 각 열의 평균과 표준편차 계산 """

def scale(data_matrix) :
    """각 열의 평균과 표준편차를 반환"""
    num_rows, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix, j))
             for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix, j))
              for j in range(num_cols)]
    return means, stdevs

print(scale(data))


""" 2. 계산된 평균과 표준편차를 이용해, 표준정규분포를 이용한 행렬 만들기"""

# 각 차원의 평균을 0, 표준편차를 1로 변환하여 척도를 조절
def rescale(data_matrix) :
    """각 열의 평균을 0, 표준편차를 1로 변환하면서
    입력되는 데이터의 척도를 조절
    편차가 없는 열은 그대로 유지"""
    means, stdevs = scale(data_matrix)

    def rescaled(i, j) :
        if stdevs[j] > 0 :
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else :
            return data_matrix[i][j]

    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescaled)

print(rescale(data))

