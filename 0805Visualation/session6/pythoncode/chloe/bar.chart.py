f=open("HR_comma_sep.csv", 'r', encoding = "UTF-8")
lines=f.readlines()
# print(lines)
number_project=[]
for line in lines :
    new_element = line.split(',')[2]
    row = number_project.append(new_element)

del row[0]

int_number_project=[]
for string in row:
    integerized=int(string)
    row = int_number_project.append(integerized)

print(row)


# from collections import Counter
# number=Counter(number_project)
# print(number)
# number_keys=list(number.keys())
# number_values=list(number.values())
# x=range(2,7)
#
# from matplotlib import pyplot as plt
# plt.bar(x, number_values)
# plt.show()
