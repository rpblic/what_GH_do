from konlpy.tag import *
twit= Twitter()
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os
os.chdir("C:\\Studying\\GrowthHackers\\0902 NLP")

moon= open('.\\wordcloud\\Moon.txt', 'rt')
ahn= open('.\\wordcloud\\Ahn.txt', 'rt')

def merge(list1, list2):
    list1_tag= [i[0] for i in list1]
    list2_tag= [j[0] for j in list2]

    result_list= []

    for key1 in list1_tag:
        key1_idx_1= list1_tag.index(key1)
        # if            여기서부터 다시 작성

moon_sent= moon.read().splitlines()
moon_list=[]

for sentence in moon_sent:
    moon_list.append(twit.nouns(sentence))
# moon_list: 리스트들의 리스트 형태, 첫 리스트는 문장 별로, 두 번째 리스트는 명사 별로

moon_nouns= list(sum(moon_list, []))        #리스트들을 하나로 합침
moon_tf= Counter(moon_nouns)
moon30word= moon_tf.most_common(30)
print(moon30word)
################################################################
ahn_sent= ahn.read().splitlines()
ahn_list=[]
for sentence in ahn_sent:
    ahn_list.append(twit.nouns(sentence))
ahn_nouns= list(sum(ahn_list, []))
ahn_tf= Counter(ahn_nouns)
ahn30word= ahn_tf.most_common(30)
print(ahn30word)

moon.close()
ahn.close()
