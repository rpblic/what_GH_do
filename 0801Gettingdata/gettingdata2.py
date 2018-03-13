# import sys, re
#
# regex= sys.argv
# print(regex)

# import json
# serialized = """ { "title" : "Data Science Book",
# "author" : "Joel Grus",
# "publicationYear" : 2014,
# "topics" : [ "data", "science", "data science"] } """
#
# deserialized = json.loads(serialized)
#
# print(type(serialized))
# print(type(deserialized))

import json, requests
from collections import Counter
from dateutil.parser import parse

#API 가져오고 직렬화하기
endpoint = "https://api.github.com/users/joelgrus/repos"
repos = json.loads(requests.get(endpoint).text)
#requests.get 메서드의 출력값은 requests.models.request이고, request.text의 출력값은 string 값이다.
# print(type(requests.get(endpoint)))
print(repos)

# Parsing dates: url에서 보낸 데이터를 그 형식에 따라 정리 또는 추출하는 과정
# dateutil.parser.parse: string으로 되어 있는 시간 데이터를 datetime.datetime 형식으로 parsing
dates = [parse(repo["created_at"]) for repo in repos]   #created_at 데이터만 추출
print(dates)

#월별, 요일별 저장소 생성 개수 알아보기
month_counts = Counter(date.month for date in dates)
weekday_counts = Counter(date.weekday() for date in dates)
print(month_counts, weekday_counts)

last_5_repositories = sorted(repos, key=lambda r: r["created_at"], reverse=True)[:5]

last_5_languages = [repo["language"] for repo in last_5_repositories]

print(last_5_languages)
