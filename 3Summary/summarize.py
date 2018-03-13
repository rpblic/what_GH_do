#-*- coding: utf-8 -*-
from funcs import *
import os
os.chdir("C:\\Users\\심규민\\Studying\\GrowthHackers\\3Summary") # 이 부분을 사용하시는 폴더 디렉토리로 수정해 주세요.
def open_and_analyze(filename):
    """데이터 파일을 열고 분석합니다."""
    '''파일을 엽니다.'''
    global takefile, docs
    takefile= open(filename, mode='r', encoding= 'UTF8')
    docs= takefile.readlines()

    '''docs 안의 문장들에 대해 유사성 검사를 실시하여 matrix로 저장합니다.'''
    matx= []
    for i, sentence in enumerate(docs):
        matx.append([])
        for j in range(len(docs)):
            lsti, lstj = sentence.split(), docs[j].split()
            matx[i].append(jacc_index(lsti, lstj))
        matx[i][i]= 0
    matx= numpy.matrix(matx)

    '''Pagerank 알고리즘을 실행하고 list로 가져옵니다.'''
    anlz= pagerank(len(matx), matx)
    anlz= anlz.tolist()
    anlz=[j[0] for i, j in enumerate(anlz)]

    '''유사성이 가장 높은 세 문장을 가져옵니다.
    (이 때 인코딩 문제로 encoding, decoding을 실시합니다.)'''
    result= bytes()
    order= g_t_k_indicies(anlz, 3)
    order.sort()
    for i in order:
        result += docs[i].encode('utf-8')

    return result.decode('utf-8')

print(open_and_analyze('input-english.txt')) # 파일명은 달라질 수 있습니다.
# 형태소 처리가 되어 있는 예제 파일에 사용하여야 올바른 분석결과가 나옵니다.
