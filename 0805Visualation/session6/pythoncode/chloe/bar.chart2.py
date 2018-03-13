f=open("HR_comma_sep.csv", 'r', encoding = "UTF-8")
lines=f.readlines()
print(lines)

"""csv파일에서 필요한 자료 추출하기"""

dept=[]
for line in lines :  # lines는 line으로 이루어진 list
    new_element = line.split(',')[8]  # 각 line을 ','로 쪼개고 그것 중 8번째(dept)
    dept.append(new_element)  # append는 list의 함수
del dept[0]  # dept의 첫번재 인수는 'sales'
print(dept)



"""변량과 도수 구하기"""
from collections import Counter # Counter는 리스트 내 각 elements 의 개수 알려줌
num_dept=Counter(dept)
print(num_dept) # Counter()는 dictionary 자료형과 형태 유샤
dept_keys=list(num_dept.keys())
dept_values=list(num_dept.values())
# python3는 a.keys() 또는 a.values()를 list로 만들 때 따로 list()로 묶어 줘야 함.
print(dept_keys)
print(dept_values)

"""그래프 그리기"""

from matplotlib import pyplot as plt
objects = dept_keys
x = [i+0.3 for i, _ in enumerate(dept_keys)] #enumerate()는 index와 인수 쌍의 나열
freq = dept_values
plt.bar(x, freq, 0.5) # 막대 그래프 그리기! plt.bar(x값, y값, width)

"""그래프 속성 추가"""

plt.xticks(x, objects) # x축에 변량명 넣어주기 plt.xticks(인덱스, 변량 이름)
plt.title("Depts")
plt.ylabel("# of people in depts")
plt.show() # 항상 마지막에! 중간에 넣으면 이후 입력된 속성 적용X

