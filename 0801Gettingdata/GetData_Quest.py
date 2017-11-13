from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import defaultdict
import re

p= re.compile(r'([가-힣]+)')
review_dict= defaultdict(list)

for j in range(10):
    html= 'http://movie.naver.com/movie/point/af/list.nhn?target=after&page='+str(j)
    html= urlopen(html)

    soup= BeautifulSoup(html, 'html5lib')
    anlyz= soup.find_all('tr')
    anlyz_list= [anlyz[i].text for i in range(1, 11)]
    # print(anlyz_list)

    for i, txt in enumerate(anlyz_list):
        title= p.findall(anlyz_list[i])[0]
        review= ' '.join(p.findall(anlyz_list[i])[1:-1])
        review_dict[title].append(review)

print(review_dict)
