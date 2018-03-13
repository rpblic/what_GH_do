"""10.2 정제하고 합치기"""

# 보통 데이터를 사용하기 전에는 정제하는 과정을 거쳐야 한다

""" 1. csv.reader를 치환하는 함수 만들기 """

def parse_row(input_row, parsers) :
    """파서 List(None이 포함될 수 도 있다)가 주어지면
    각 input_row의 항목에 적절한 파서를 적용"""
    return [parser(value) if parser is not None
            else value for value, parser in zip(input_row, parsers)]

def parse_rows_with(reader, parsers) :
    """각 열에 파서를 적용하기 위해 reader을 치환"""
    for row in reader :
        yield parse_row(row, parsers)

"""  2. 나쁜 데이터가 포함되어 있을 때 보완하기 """

def try_or_none(f) :
    """f가 하나의 입력값을 받는다고 가정하고,
    오류가 발생하면 f는 None을 반환해주는 함수로 치환하자"""
    def f_or_none(x) :
        try : return f(x)
        except : return None
    return f_or_none

def parse_row_new(input_row, parsers) :
    return [try_or_none(parser)(value) if parser is not None
            else value for value, parser in zip(input_row, parsers)]

def parse_rows_with_new(reader, parsers) :
    """각 열에 파서를 적용하기 위해 reader을 치환"""
    for row in reader :
        yield parse_row_new(row, parsers)

import csv
import codecs

data = []

with codecs.open("HR_comma_sep.csv","rb","utf-8") as f :
    reader = csv.reader(f)
    for line in parse_rows_with_new(reader, [ None, None, None, None, None, None, None, None, None, None ]) :
        data.append(line)

print(data)
"""
for row in data :
    if any(x is None for x in row) :
        print(row)
"""

"""
#3. csv.DictReader에 대한 헬퍼함수

 def try_parse_field(field_name, value, parser_dict) :
   #parse_dict에 포함되어 있는 파서 중에 하나로 파싱
    parser = parser_dict.get(field_name)
    if parser is not None :
        return try_or_none(parser)(value)
    else :
        return value

def parse_dict(input_dict, parser_dict) :
    return {field_name : try_parse_field(field_name, value, parser_dict)
            for field_name, value in input_dict.iteritems()}
"""