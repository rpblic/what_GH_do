import gmplot
from bs4 import BeautifulSoup 
import urllib
import matplotlib.pyplot as plt
import numpy as np


def parse_result(inputText):
    event_id = []
    origin_time = []
    evla = []
    evlo = []
    evdp = []
    mag = []
    mag_type = []
    EventLocationName  = []
    for i, item in enumerate(inputText.split('\n')[0:-1]):
        if i < 1:
            continue

        try:
            splited = item.split('|')
            event_id.append(splited[0])
            origin_time.append(splited[1])
            evla.append(splited[2])
            evlo.append(splited[3])
            evdp.append(splited[4])
            mag.append(splited[10])
            mag_type.append(splited[9])
            EventLocationName.append(splited[-1])
        except:
            print item
            print 'something wrong'

    return np.c_[event_id, origin_time, evla, evlo, mag, mag_type, EventLocationName]

url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=text&starttime=2010-01-01&endtime=2016-01-01&minmagnitude=5.0'

r = urllib.urlopen(url).read()
soup = BeautifulSoup(r) #
events_mat = parse_result(soup.text)

lats = [float(item[2]) for item in events_mat]
lons = [float(item[3]) for item in events_mat]






gmap = gmplot.GoogleMapPlotter(0, 0, 2)
gmap.heatmap(lats, lons)
gmap.draw("3_my_heatmap.html")
