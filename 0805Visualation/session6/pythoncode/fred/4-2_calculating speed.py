from math import *
import matplotlib.pyplot as plt
f=open('./gpsdata.txt','r')
lines=f.readlines()
switch=0

speedlist = []
for line in lines:
    if switch==0:
        data0=line.strip().split('\t')
        switch=1
    else:
        data1=line.strip().split('\t')
        
        a = (float(data0[1])-float(data1[1]))/pow(10,7)*pi/180
        b = (float(data0[2])-float(data1[2]))/pow(10,7)*pi/180
        c = abs(acos(cos(a)*cos(b))*6400000)
        intertime = (float(data0[3]) - float(data1[3])) /1000.
        data0 = data1
        if intertime > 0:
            speed = c / intertime
            if speed < 1000:
                speedlist.append(speed)
            


plt.hist(speedlist)
plt.yscale('log', nonposy='clip')
plt.show()