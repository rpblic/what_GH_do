from konlpy.tag import Twitter
t = Twitter()
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc



moon = open('C:/Users/LG/Desktop/Moon.txt', 'r').read()
ahn = open('C:/Users/LG/Desktop/Ahn.txt', 'r').read()


#문재인 연설문에서 명사만 뽑아 리스트로 만들기
moon_sent = moon.splitlines()
moon_list=[] #명사만 뽑아내어 이 리스트에 넣을 것!

for sentence in moon_sent:
    moon_list.append(t.nouns(sentence))

"""여기까지 하면 연설문의 한 문장 당 리스트 하나씩이 만들어져서 
moon_list는 리스트들의 리스트가 됨"""

moon_nouns = list(sum(moon_list, [])) #리스트들을 하나로 합쳐주는 과정


#안철수 연설문에서 명사만 뽑아 리스트로 만들기
ahn_sent = ahn.splitlines()
ahn_list=[]

for sentence in ahn_sent:
    ahn_list.append(t.nouns(sentence))

ahn_nouns = list(sum(ahn_list, []))
####################################################


#각 후보가 가장 많이 사용한 명사만 30개씩 추리기
moon_tf = Counter(moon_nouns)
ahn_tf = Counter(ahn_nouns)
moon30word=moon_tf.most_common(30)
ahn30word=ahn_tf.most_common(30)
print(moon30word)
print(ahn30word)


"""여기까지 하면 

moon30word = [('국민', 32), ('저', 18), ('대통령', 13), .... ]
ahn30word = [('국민', 36), ('안철수', 15), ('대통령', 11), ....]

형태가 만들어지는데, 
각 명사가 두 연설문에 등장한 횟수를 비교할 수 있도록 

[('국민', 32, 36), ('저', 18, 0), ('대통령', 13, 11), ('안철수', 0, 15), ... ]

이런 형태로 만들고자 merge함수를 정의합니다 (아래)
"""
def merge(list1, list2):
    list1_tag = [i[0] for i in list1]
    list2_tag = [j[0] for j in list2]

    result_list = []

    for key1 in list1_tag:
        key1_idx_1 = list1_tag.index(key1)
        if key1 in list2_tag:
            key1_idx_2 = list2_tag.index(key1)
            result_list.append((list1[key1_idx_1][0], list1[key1_idx_1][1], list2[key1_idx_2][1]))
        else:
            result_list.append((list1[key1_idx_1][0], list1[key1_idx_1][1], 0))

    for key2 in list2_tag:
        key2_idx_2 = list2_tag.index(key2)
        if key2 not in list1_tag:
            result_list.append((list2[key2_idx_2][0], 0, list2[key2_idx_2][1]))
    return result_list


""" 정의한 함수로 mergelist 만들기"""
mergelist = merge(moon30word, ahn30word)



""" 책에서 사용한 방법 그대로 워드클라우드 만들기"""
def text_size(total):
    return 8 + total / 10

for word, moon_speech, ahn_speech in mergelist :
    plt.text(moon_speech, ahn_speech, word, ha='center', va='center', size=text_size(moon_speech + ahn_speech))

font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.xlabel("문재인", fontproperties=font_name)
plt.ylabel("안철수", fontproperties=font_name)
plt.axis([0, 40, 0, 40])
plt.xticks([])
plt.yticks([])
plt.show()