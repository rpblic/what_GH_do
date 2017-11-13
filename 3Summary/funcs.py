from collections import defaultdict as ddict
from collections import Counter
from math import sqrt
import numpy

def sort_indicies(arr):
    """Rank를 오름차순으로 Sorting합니다."""
    '''Array의 속성 검사'''
    for i in range(1, len(arr)):
        if type(arr[i])!= int: print("Error"); return None
        for j in range(i):
            if arr[i]==arr[j]: print("Error"); return None

    '''Sorting'''
    return sorted(arr, reverse=False)

def g_t_k_indicies(arr, howmany):
    """List에서 연관도가 가장 높은 세 값의 order를 가져옵니다."""
    maxarr= sorted(arr, reverse=True)[:howmany]
    return [arr.index(i) for i in maxarr]

def frob_norm(matx):
    """matx의 각 데이터값을 제곱하고 제곱근합니다."""
    rtn= 0
    for i in range(len(matx)):
        for j in range(len(matx[i])):
            rtn += matx[i][j]**2
    return sqrt(rtn)

def jacc_index(lst1, lst2):
    """두 개의 list를 multiset(counter)로 변환하고 Jaccard index를 계산합니다."""
    return sum((Counter(lst1)&Counter(lst2)).values())/sum((Counter(lst1)|Counter(lst2)).values())

# print(type(numpy.matrix(numpy.zeros((3,3)))))
def pagerank(n, matx):
    """TextRank를 계산합니다. 행렬에서!"""
    '''변수를 설정합니다.'''
    npmatx= numpy.matrix(matx, dtype= 'float32')
    vec_r= numpy.ones((n, 1))/ n
    vec_p= numpy.ones((n, 1))
    const_d= 0.85
    matxsum= numpy.sum(npmatx, axis= 1) # M 행렬 계산을 위해

    '''M 행렬을 만듭니다.'''
    for i in range(n):
        # print(npmatx[i, :], matxsum[i, 0])
        npmatx[i, :]= npmatx[i, :]/ matxsum[i, 0]
    matx_m= npmatx.T

    '''M 행렬의 각 행에서의 TextRank를 구합니다.'''
    while True:
        # print(vec_r)
        vec_rr= const_d*matx_m*vec_r + (1-const_d)/n*vec_p
        if frob_norm(vec_rr-vec_r) < 0.000001:
            return vec_rr
        vec_r= vec_rr

# matx= [[0,1,1,0], [0,0,0,1], [1,1,0,1], [0,0,1,0]]
# print(pagerank(len(matx), matx))
