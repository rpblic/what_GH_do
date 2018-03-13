# ps= 5/14; po= 4/14; pr= 5/14
# psy= 3/5; poy= 1; pry= 3/5
import os
os.chdir('C:\\Studying\\GrowthHackers\\0822 Decision')
# from 170822_decision_tree import *
import math
# print(-(\
#     ps*(psy*math.log2(psy) + (1-psy)*math.log2(1-psy))+\
#     po*(poy*math.log2(poy)) + \
#     pr*(pry*math.log2(pry) + (1-pry)*math.log2(1-pry))))

# 실습 코드

# 1) 전체 엔트로피 계산 & 각 attribute로 파티션한 후 엔트로피 계산

print('inputs', format(data_entropy(inputs), '.4f'))
for key in ['Outlook','Temperature','Humidity','Windy']:
    print(key, format(partition_entropy_by(inputs, key), '.4f'))
print()
"""Outlook의 엔트로피가 가장 낮으며, Outlook으로 파티션 했을 경우의 Information gain이 가장 높다"""

# 2) Outlook으로 파티션 한 뒤 그 안에서 Sunny로 또 파티션 하기
Out_Sunny_inputs = [(input, label)
                     for input, label in inputs if input["Outlook"] == "sunny"]
print(Out_Sunny_inputs)
print('Out_Sunny_inputs', format(data_entropy(Out_Sunny_inputs), '.3f'))
for key in ["Temperature", "Humidity", "Windy"]:
    print(key, partition_entropy_by(Out_Sunny_inputs, key))
print()

"""Outlook으로 파티션 한 후 다시 Sunny로 파티션 하고,
   그 다음 각 attriute로 파티션을 해 보았더니 Humidity로 파티션 한 결과의 엔트로피가 가장 낮았다"""

Out_Sunny_Humid_inputs = [(input, label)
                            for input, label in Out_Sunny_inputs if input["Humidity"] == 'high']
print(Out_Sunny_Humid_inputs)
print('Out_Sunny_Humid_inputs', format(data_entropy(Out_Sunny_Humid_inputs), '.3f'))
for key in ["Temperature", "Windy"]:
    print(key, partition_entropy_by(Out_Sunny_Humid_inputs, key))
print()

# 3) Outlook에서 overcast 선택 시
Out_Over_inputs = [(input, label)
                        for input, label in inputs if input["Outlook"] == "overcast"]
print(Out_Over_inputs)
print('Out_Over_inputs', format(data_entropy(Out_Over_inputs), '.3f'))
print()

# 4) Outlook에서 rain 선택 시
Out_Rain_inputs = [(input, label)
                       for input, label in inputs if input["Outlook"] == "rain"]
print(Out_Rain_inputs)
print('Out_Rain_inputs', format(data_entropy(Out_Rain_inputs), '.3f'))
for key in ["Temperature", "Humidity", "Windy"]:
    print(key, partition_entropy_by(Out_Rain_inputs, key))
print()


# 5) 전체 의사결정 나무 그리기
print("building the tree")
tree = build_tree_id3(inputs)
print(tree)
print()

# 6) 새로 입력된 하나의 데이터 값을 의사결정 나무에 따라 분류해보기
print("sunny / mild / high / TRUE", classify(tree,
        { "Outlook" : "sunny",
          "Temperature" : "mild",
          "Humidity" : "high",
          "Windy" : "TRUE"} ))
