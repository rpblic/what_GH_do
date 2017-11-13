import os
os.chdir("C:\\Studying\\GrowthHackers\\0805Visualation\\session6")

from matplotlib import pyplot as plt
from collections import Counter
with open("HR_comma_sep.csv", "rt") as dataopen:
    lines= dataopen.readlines()
    splitedlines= [line.split(',') for line in lines]
    # index= splitedlines[0]
    # print(index)
    # print(splitedlines[0])
    # salelist= [splitedlines[i][8] for i in range(len(splitedlines))]
    evallist_str= [splitedlines[i][1] for i in range(len(splitedlines))]
    del evallist_str[0]
    evallist= list(map(float, evallist_str))
    print(len(evallist_str), len(evallist))
    # print(evallist[:10])
    # salecounter= Counter(salelist)
    evalcounter= Counter(evallist)

    # print(salarycounter['sales'])
    k= (max(evalcounter.keys())-min(evalcounter.keys()))/20
    func= lambda x: ((x-min(evalcounter.keys()))//k)*k + min(evalcounter.keys())
    pltindex= Counter(func(x) for x in evallist)
    print(pltindex)

    loc= [i+0.3 for i, _ in enumerate(pltindex)]
    frequency= list(pltindex.values())
    plt.bar(loc, frequency, 0.5)
    plt.xticks(loc, pltindex.keys())
    plt.ylabel("# of people in depts")
    plt.title("Depts")
    plt.show()
