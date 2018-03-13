f=open("HR_comma_sep.csv", 'r', encoding = "UTF-8")
lines=f.readlines()
print(lines)

# # csv파일에서 필요한 자료 추출하기 (1)
# satisfaction=[]
# last_evaluation=[]
# for line in lines : #lines는 line으로 이루어진 list
#     ele_satisfaction = line.split(',')[0] #각 line을 ','로 쪼개고 그것 중 0번째(dept)
#     satisfaction.append(ele_satisfaction) #append는 list의 함수
#     ele_evaluation = line.split(',')[1]
#     last_evaluation.append(ele_evaluation)
#
# del satisfaction[0] # dept의 첫번재 인수는 'sales'
# del last_evaluation[0]
# print(satisfaction)
# print(last_evaluation)
#
# # 문자 list를 숫자 list로! / python3는 list() 따로 묶어줘야 함!
# satisfation = list(map(float,satisfaction))
# last_evaluation = list(map(float, last_evaluation))

"""csv파일에서 필요한 자료 추출하기"""

satisfaction=[]
tsc=[]
for line in lines : #lines는 line으로 이루어진 list
    ele_satisfaction = line.split(',')[0] #각 line을 ','로 쪼개고 그것 중 0번째(dept)
    satisfaction.append(ele_satisfaction) #append는 list의 함수
    ele_tsc = line.split(',')[3] #tsc=time_spend_company
    tsc.append(ele_tsc)

del satisfaction[0]
del tsc[0]

#문자 list를 숫자 list로! / python3는 list() 따로 묶어줘야 함!
satisfation = list(map(float,satisfaction))
tsc = list(map(int, tsc))



# 그래프 그리기 (+꾸미기)
from matplotlib import pyplot as plt
plt.scatter(satisfaction, tsc, s=2, color='red', marker='s')
plt.grid(True)
plt.title('Is there any Correlation?')
plt.xlabel("satisfaction_level")
plt.ylabel("time_spend_company")
plt.show()

# plt.axis()


plt.show() # 어느 일정 수준에 도달하면 만족도와 지난 직무 평가결과와는 관련이 없다?



