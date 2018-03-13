import csv
from bs4 import BeautifulSoup

reader = csv.reader(open('7_seoul_dentist.txt', 'r'), delimiter=',')
svg = open('7_seoul_edit.svg', 'r').read()
min_value = 100; max_value = 0; past_header = False
dentist_count = {}
count_only = []
for row in reader:
    if not past_header:
        past_header = True
    try:
        unique = row[0]
        count = float(row[1].strip())
        dentist_count[unique] = count
        count_only.append(count)
    except:
        pass

soup = BeautifulSoup(svg)

paths = soup.find_all('g') # or FindAll

colors = ["#CCE0FF", "#99C2FF", "#66A3FF", "3385FF", "0066FF", "0052CC", "#003D99"]
g_style = 'fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'
#print dentist_count
for p in paths:
    #print p
    try:
        count = dentist_count[p['id']]
    except:
        continue
    if count > 70:
        color_class = 6
    elif count > 55:
        color_class = 5
    elif count > 45:
        color_class = 4
    elif count > 25:
        color_class = 3
    elif count > 15:
        color_class = 2
    elif count > 10:
        color_class = 1
    else:
        count_class = 0
    color = colors[color_class]
    p['style'] = g_style + color
f = open('test5.svg', 'w')
f.write(soup.prettify())
f.close()
