from collections import Counter, defaultdict
from functools import partial
import math, random

""" 17.2 엔트로피"""

"""엔트로피
불순도(impurity) 혹은 불확실성(uncertainty)
엔트로피의 감소는 불순도 혹은 불확실성이 감소,
순도(homogeneity)가 증가. 이를 정보이론에서는 정보획득(information gain)이라고 함. 
결국 엔트로피는 어떤 데이터가 균일한 정도를 나타내는 지표, 즉 순도를 계산하는 한 방식"""

# 1
def entropy(class_probabilities):
    """어떤 레이블에 속할 확률을 입력하면 엔트로피를 계산하는 공식"""
    return sum(-p * math.log(p, 2) for p in class_probabilities if p)
    # if 조건문 중 '숫자'자료형: 숫자가 0이 아니면 True, 0이면 False
    # 즉 여기서는 확률이 0인 경우는 제외한 것이다

# 2
def class_probabilities(labels):
    """어떤 레이블에 속할 확률을 구함"""
    total_count = len(labels) # labels 라는 전체 데이터 리스트에서, 데이터의 총 개수 구함
    return [count / total_count
            for count in Counter(labels).values()]
    # 데이터들이 각 레이블에 속할 확률을 리스트로 반환
    # Counter: element는 key로, element들의 count횟수는 value로 포함하는 dict를 만든다
    # values() : dict의 method. value의 리스트를 반환
    # Counter(labels).values() 의 결과는 {'label1' : 'label1 count 횟수', 'label2' : 'label2 count 횟수'}
    # 결과값이 이진인 경우만 고려, 그러므로 label의 종류에는 label1(True), label2(False) 두개

# 3
def data_entropy(labeled_data):
    """데이터 전체에 대한 엔트로피 계산"""
    labels = [label for _, label in labeled_data]
    #labeled_data는 (input, label)쌍으로 구성됨. 그 중 label만 갖고 와서 labels 리스트를 만듦
    probabilities = class_probabilities(labels)
    # 위에서 만든 labels 리스트, 즉 전체 데이터에 대해 엔트로피 구함.
    return entropy(probabilities)

""" 17,3 파티션의 엔트로피"""

# 4
def partition_entropy(subsets):
    """subsets는 레이블이 있는 데이터 list의 list이다. 여기서는 subsets이 전체 집합에 해당
    그에 대한 파티션 엔트로피, 즉 부분집합의 엔트로피를 구하라"""
    total_count = sum(len(subset) for subset in subsets)
    # 각 subset들에 포함된 데이터들의 개수를 모두 더해야 총 데이터의 개수인 total_count를 구할 수 있다
    # subsets = [ [subset 1], [subset 2], ... , [마지막 subset] ] 이런 모양이다

    return sum( data_entropy(subset) * len(subset) / total_count
                for subset in subsets )
    # 각 subset의 엔트로피 구한 후 해당 subset에 속할 확률을 곱해주어
    #  파티션 후 가중평균시킨 엔트로피의 값을 구함
    # 이 값은 기존의, 분할 이전 데이터 전체에 대해 구한 엔트로피 값보다 낮을 것이다

""" 17.4 의사 결정 나무 """

"""
ID3 알고리즘 (범주형 자료만 분류 가능) -> 코드쓸 때 필요한 개념
- Iterative Dichotomiser 3, (C4.5의 전신이다)
1. 루트노드 생성
2. 현재 트리에서 모든 단말 노드에 대해서 아래를 반복한다.
 A. 해당 노드의 샘플들이 같은 클래스이면, 해당 노드는 단말노드가 되고, 해당 클래스로 레이블을 부여
 B. 더 이상 사용할 수 있는 속성이 없으면 수행 종료
 C. Information Gain이 높은 속성을 선택해서 노드를 분할
 
-‘greedy(탐욕적)’ 알고리즘
1. 모든 데이터 포인트의 클래스 레이블(True/False)이 동일하다면, 
    그 예측값이 해당 클래스 레이블인 잎 노드를 만들고 종료하라.
2.
1) 파티션을 나눌 수 있는 변수(attribute)가 남아있지 않다면(즉, 더 이상 물을 수 있는 질문이 없다면), 가장 빈도수가 높은 클래스 레이블로 예측하는 잎 노드를 만들고 종료하라
2) 그게 아니면 각 변수로 데이터의 파티션을 나누어라
3) 파티션을 나눴을 때 엔트로피가 가장 낮은 변수를 택하라
4) 선택된 변수에 대한 결정 노드를 추가하라
5) 남아 있는 변수들로 각 파티션에 대해 위 과정을 반복하라
"""

# 5
def group_by(items, key_fn):
    """함수 key_fn(item)의 결과를 key 값으로 하고 item들이 포함된 list를 value로 하는
    groups라는 이름의 dictionary를 반환함"""
    groups = defaultdict(list) # value값이 기본적으로 빈 list의 형식을 가지는 groups라는 dictionary
    for item in items: # 주어진 items라는 iterable한 데이터의 각 원소에 대해
        key = key_fn(item) # 함수 key_fn(item) 의 결과를 구하여 key라고 둔다.
        groups[key].append(item)
        # groups 리스트에
        # 위에서 구한 결과값 key를 key로 두고, 그에 대응되는 value 리스트에 item을 추가시킨다.
    return groups

# 6 - 한국 책 버전
def partition_by(inputs, attribute):
    """attribute에 따라 inputs의 파티션을 나누자"""
    # inputs 모양 : [ ({key1 : value1, key2 : value2 }, True), ({key1 : value1, key2 : value2}, False) ]
    # input 모양 : ({key : value}, True)
    groups = defaultdict(list) # value값이 기본적으로 빈 list의 형식을 가지는 groups라는 dictionary
    for input in inputs: # 주어진 iterable한 데이터, inputs의 각 원소인 input에 대해
        key = input[0][attribute]
        # 각 input의 [0]번째 원소인 딕셔너리에서 attriute라는 키에 해당하는 value를 'key'로 정의
        groups[key].append(input)
        # groups 라는 딕셔너리에서,
        # 위에서 구한 결과값 key를 key로 두고, 그에 대응되는 value 리스트에 input을 추가시킨다.
    return groups # groups 라는 딕셔너리 반환

# 7
def partition_entropy_by(inputs,attribute):
    """주어진 attribute로 파티션(부분집합)을 나눈 뒤 각 파티션(부분집합)의 엔트로피 계산"""
    partitions = partition_by(inputs, attribute)
    # 주어진 attribute로 partition_by을 사용해 부분집합을 나눔.
    # 그 결과는 groups라는 딕셔너리인데, 이름을 partitions로 바꾼 후
    return partition_entropy(partitions.values())
    # partitions 딕셔너리의 각 key값들의 value, 즉 리스트로 표현된 각 부분집합들의 entropy를 구한다

""" 17.5 종합하기 """

# 8.
def classify(tree, input):
    """ 주어진 입력값 input을 의사결정나무 tree를 이용하여 분류하자 """
    # input 모양 : ({key : value}, True/False)

    # 잎 노드, 즉 나무 자체가 바로 True나 False 값을 갖고 있으면 그 값을 반환
    if tree in [True, False]:
        return tree

    # 그게 아니라면 데이터의 변수(attribute)로 파티션을 나누자
    # 키로 변수 값, 값으로 서브트리를 나타내는 dict를 사용하면 된다
    attribute, subtree_dict = tree # tree는 (attribute, subtree_dict)의 tuple 형식이다

    subtree_key = input.get(attribute)
    # 위의 tree (attribute, subtree_dict)에서
    # 주어진 변수(attribute)를 key로 하는 value값을 subtree_key로 둔다.

    if subtree_key not in subtree_dict:
        subtree_key = None
        # subtree_key에 해당하는 서브트리(subtree_dict)가 존재하지 않을 때
        # None 서브트리를 사용
        # 즉 입력된 데이터의 변수 중 하나가 기존에 관찰되지 않았다면 None이 된다.

    subtree = subtree_dict[subtree_key]
    return classify(subtree, input)
    # 위에서 정의한 subtree_key를 사용해 적절한 서브트리(subtree_dict)를 선택
    # 그리고 입력된 데이터를 계속, 끝까지 분류 - 재귀적(recursive)!

# 9
def build_tree_id3(inputs, split_candidates=None):
    """학습용 데이터로부터 실제 나무를 구축하기"""
    # 만약 파티션하는 첫 번째 단계라면
    # 입력된 데이터의 모든 변수가 파티션 기준 후보
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()
    # dict.keys()는 딕셔너리의 키값만 모아서 dict_keys 개체를 반환한다
    # dict_keys(['Outlook', 'Temperature', 'Humidity', 'Windy'])

    # 입력된 데이터에서 True와 False의 개수를 세어 본다
    # inputs 모양 : [ ({key1 : value1, key2 : value2 }, True), ({key1 : value1, key2 : value2}, False) ]
    # 전체 데이터 inputs 안의 각 tuple이 개별 데이터 input
    num_inputs = len(inputs) # 개별 데이터의 갯수
    num_trues = len([label for item, label in inputs if label])
    # if 조건문 : 참/ 거짓 판단 후 참일 시 앞의 작업 시행
    # 여기서는 label이 True이면 앞의 작업 시행
    # 즉, 해당 True를 리스트에 추가하고, 후에 리스트 내의 True의 총 갯수를 센다
    num_falses = num_inputs - num_trues

    if num_trues == 0:                  # True가 하나도 없다면 False 잎을 반환
        return False

    if num_falses == 0:                 # False가 하나도 없다면 True 잎을 반환
        return True

    if not split_candidates:            # 만약 파티션 기준으로 사용할 변수가 없다면
        return num_trues >= num_falses  # 다수결로 결과를 결정

    # 아니면 가장 적합한 변수를 기준으로 파티션
    best_attribute = min(split_candidates,
        key=partial(partition_entropy_by, inputs))
    # partial(func, argu) 메소드는 func(argu)를 실행해준다
    # partition_entropy_by(inputs, attribute) :
    # 주어진 attribute로 파티션(부분집합)을 나눈 뒤 각 파티션(부분집합)의 엔트로피 계산
    # partition_entropy_by(inputs, split_candidates)의 최소값을 만드는
    # split_candidate가 best_attribute

    partitions = partition_by(inputs, best_attribute)
    # inputs를 best_attribute에 따라 파티션을 나눔. partitions는 dict 자료형
    new_candidates = [a for a in split_candidates
                      if a != best_attribute]
    # new_candidates는 앞서 best_attribute가 아니였던 attribute들을 모아놓은 리스트

    # 재귀적으로 서브트리를 구축
    subtrees = { attribute : build_tree_id3(subset, new_candidates)
                 for attribute, subset in partitions.items() }
    # dict.item은 딕셔너리의 키와 밸류를 모아둔 튜플을 가진
    # dict_items([('key1', 'value1'), ('key2', 'value2')]) 객체를 반환
    # partition.items 객체의 (attribute, subset)에서 attribute를 가져와서
    # attribute를 키로 하고, build_tree_id3(subset, new_candidates)의 결과값을 value로 하는
    # subtrees 라는 dict를 만듦
    subtrees[None] = num_trues > num_falses
    # subtrees에서 "None"이라는 key값의 value는 다수결에 따라 True, False 결정

    return (best_attribute, subtrees)
    # (best_attribute, subtrees)라는 tuple을 반환

""" 17.6 랜덤 포레스트 """

# 랜덤 포레스트: 여러개의 의사결정 나무를 만들고, 그들의 다수결로 결과를 결정하는 방법
def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]

""" 17.7 예제 """
# 0) 전체 데이터 "inputs"

inputs = [
        ({'Outlook':'sunny','Temperature':'hot','Humidity':'high','Windy':'FALSE'}, False),
        ({'Outlook': 'sunny', 'Temperature': 'hot', 'Humidity': 'high', 'Windy': 'TRUE'}, False),
        ({'Outlook': 'overcast', 'Temperature': 'hot', 'Humidity': 'high', 'Windy': 'FALSE'}, True),
        ({'Outlook': 'rain', 'Temperature': 'mild', 'Humidity': 'high', 'Windy': 'FALSE'}, True),
        ({'Outlook': 'rain', 'Temperature': 'cool', 'Humidity': 'normal', 'Windy': 'FALSE'}, True),
        ({'Outlook': 'rain', 'Temperature': 'cool', 'Humidity': 'normal', 'Windy': 'TRUE'}, False),
        ({'Outlook': 'overcast', 'Temperature': 'cool', 'Humidity': 'normal', 'Windy': 'TRUE'}, True),
        ({'Outlook': 'sunny', 'Temperature': 'mild', 'Humidity': 'high', 'Windy': 'FALSE'}, False),
        ({'Outlook': 'sunny', 'Temperature': 'cool', 'Humidity': 'normal', 'Windy': 'FALSE'}, True),
        ({'Outlook': 'rain', 'Temperature': 'mild', 'Humidity': 'normal', 'Windy': 'FALSE'}, True),
        ({'Outlook': 'sunny', 'Temperature': 'mild', 'Humidity': 'normal', 'Windy': 'TRUE'}, True),
        ({'Outlook': 'overcast', 'Temperature': 'mild', 'Humidity': 'high', 'Windy': 'TRUE'}, True),
        ({'Outlook': 'overcast', 'Temperature': 'hot', 'Humidity': 'normal', 'Windy': 'FALSE'}, True),
        ({'Outlook': 'rain', 'Temperature': 'mild', 'Humidity': 'high', 'Windy': 'TRUE'}, False),
    ]
