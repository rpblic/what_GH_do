"""10.3 데이터 처리"""

""" 0. 사전 작업 """

# 1) 파일 읽기

import p10_2
import csv
import codecs

whole = []

with codecs.open("HR_comma_sep1.csv","rb","utf-8") as f :
    reader = csv.reader(f)
    for line in p10_2.parse_rows_with_new(reader, [ None, None, None, None, None, None, None, None, None, None ]) :
        whole.append(line)

data = []
keys = whole[0]
for contents in whole[1:] :
    values = contents
    dictionary = dict(zip(keys, values))
    data.append(dictionary)

print(data)


""" 1. accounting 부서의 average_montly_hours 최고치 찾기 """

max_avg_hours_company = max(row["average_montly_hours"]
                     for row in data
                     if row["sales"] == "accounting")

print(max_avg_hours_company)


""" 2. 모든 부서의 최고 average_montly_hours 찾기"""

from collections import defaultdict

# 부서(sales)을 기준으로 행을 그룹화
by_sales = defaultdict(list)
for row in data :
    by_sales[row["sales"]].append(row)

# list_comprehension으로 각 그룹의 최고치 계산
max_avg_hours_by_sales = {sales : max(row["average_montly_hours"]
                                    for row in grouped_rows)
                        for sales, grouped_rows in by_sales.items()}

print(max_avg_hours_by_sales)


"""3. dict의 특정 필드를 갖고 오는 함수/ 여러 dict에서 동일한 필드를 갖고 오는 함수"""

def picker(field_name) :
    """dict의 특정 필드를 선택해 주는 함수를 반환"""
    return lambda row : row[field_name]

def pluck(field_name, rows) :
    """dict list를 필드 리스트로 변환"""
    return map(picker(field_name), rows)

"""4. grouper 함수 : 여러 행을 하나의 그룹으로 묶어주기"""

def group_by(grouper, rows, value_transform = None) :
    # key는 grouper의 결과값이며 value는 각 그룹에 속하는 모든 행의 list
    grouped = defaultdict(list)
    for row in rows :
        grouped[grouper(row)].append(row)
    if value_transform is None :
        return grouped
    else :
        return {key : value_transform(rows)
                for key, rows in grouped.items()}

# 앞에서 max_avg_hours_by_sales를 나타내었던 것을 더욱 간단히 나타낼 수 있다
max_avg_hours_by_sales2 = group_by(picker("sales"),
                               data,
                               lambda rows : max(pluck("average_montly_hours", rows)))

print(max_avg_hours_by_sales2)
