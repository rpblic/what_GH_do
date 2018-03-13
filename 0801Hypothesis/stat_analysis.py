import numpy as np
from scipy.stats import *
from pprint import *

class stat_analysis:

    def __init__ (self, file_path): # 파일 디렉토리
        f = open(file_path, 'r', encoding = 'UTF-8')
        self.file = f.name # 파일 이름
        self.file_lines = f.readlines() # 파일 각 줄 저장
        f.close()

    def line_by_line(self): # 각 줄 사전으로 만들어 출력
        col_names = self.file_lines[0][:-1].split(',') # '\n' 제외, split
        result_list = []
        for line in self.file_lines[1:]:
            splited = line[:-1].split(',')
            row_dict = {}
            for i in range(len(splited)):
                row_dict[col_names[i]] = splited[i]
            result_list.append(row_dict)
        pprint(result_list)

    def make_comp_ndarray(self, std_comp, std_comp_value, tg_comp):
        # 특정 기준에 해당하는 값을 모아 np.ndarray 형태로 반환하는 메소드
        # std_comp : 그룹 분류 기준 항목, std_comp_value : std_comp의 기준 값
        # tg_comp : 목표 분류 항목
        # 가령 make_comp_ndarray("am", 0, "mpg")는 "am" 값이 0인 것들의 "mpg" 값들을 ndarray로
        col_comps_list = self.file_lines[0][:-1].split(',')
        s_idx = col_comps_list.index(std_comp)
        t_idx = col_comps_list.index(tg_comp)
        return_list = []
        for line in self.file_lines[1:]:
            splited = line[:-1].split(',')
            if int(splited[s_idx]) == int(std_comp_value):
                return_list.append(splited[t_idx])
        ary = np.array(return_list).astype(np.float)
        return ary

    def eq_var_test(self, std_comp, std_comp_value_1, std_comp_value_2, tg_comp):
        # scipy.stat의 fligner 함수 활용해서 std_comp 항목의 값이
        # std_comp_value_1인 것의 tg_comp 값들을 모은 그룹
        # 그리고 std_comp_value_2인 것의 tg_comp 값들을 모은 그룹 사이
        # 등분산성 검정
        ary_1 = self.make_comp_ndarray(std_comp, std_comp_value_1, tg_comp)
        ary_2 = self.make_comp_ndarray(std_comp, std_comp_value_2, tg_comp)
        return fligner(ary_1, ary_2)

    def ttest_ind(self, std_comp, std_comp_value_1, std_comp_value_2, tg_comp, Significance_Level):
        # 앞서 정의한 eq_var_test와 scipy.stat의 ttest_ind 함수 활용해서
        # std_comp 항목의 값이 std_comp_value_1인 것의 tg_comp 값들을 모은 그룹
        # 그리고 std_comp_value_2인 것의 tg_comp 값들을 모은 그룹 사이
        # 독립성 가정한 t 검정 (Significance_Level에 따라 등분산성/이분산성)
        ary_1 = self.make_comp_ndarray(std_comp, std_comp_value_1, tg_comp)
        ary_2 = self.make_comp_ndarray(std_comp, std_comp_value_2, tg_comp)
        if self.eq_var_test(std_comp, std_comp_value_1, std_comp_value_2, tg_comp).pvalue < Significance_Level:
            return ttest_ind(ary_1, ary_2, equal_var = False)
        else:
            return ttest_ind(ary_1, ary_2, equal_var = True)

    def ttest_rel(self, std_comp, std_comp_value_1, std_comp_value_2, tg_comp, Significance_Level):
        # 앞서 정의한 eq_var_test와 scipy.stat의 ttest_rel 함수 활용해서
        # std_comp 항목의 값이 std_comp_value_1인 것의 tg_comp 값들을 모은 그룹
        # 그리고 std_comp_value_2인 것의 tg_comp 값들을 모은 그룹 사이
        # 독립성 X 가정한 t 검정 (Significance_Level에 따라 등분산성/이분산성)
        ary_1 = self.make_comp_ndarray(std_comp, std_comp_value_1, tg_comp)
        ary_2 = self.make_comp_ndarray(std_comp, std_comp_value_2, tg_comp)
        if self.eq_var_test(std_comp, std_comp_value_1, std_comp_value_2, tg_comp).pvalue < Significance_Level:
            return ttest_rel(ary_1, ary_2, equal_var = False)
        else:
            return ttest_rel(ary_1, ary_2, equal_var = True)
