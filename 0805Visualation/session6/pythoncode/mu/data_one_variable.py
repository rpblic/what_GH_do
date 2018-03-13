import os
import pandas as pd
import matplotlib.pyplot as plt

os.path.abspath(os.path.curdir)
os.chdir(r'/Users/kangmina')

f=open("HR_comma_sep.csv","r",encoding="UTF-8")
lines=f.readlines()
average_montly_hours=[]
for line in lines:
	worker_amh=line.split(',')[2]
	average_montly_hours.append(worker_amh)
print(average_montly_hours)

del average_montly_hours[0]
int_average_monthly_hours=list(map(int,average_montly_hours))

print(int_average_monthly_hours)

pd.Series(int_average_monthly_hours)
seriesform=pd.Series(int_average_monthly_hours)
print(seriesform)

seriesform.describe()


plt.style.use('ggplot')
plt.hist(seriesform, bins=20, alpha=0.5, color='c')
plt.axvline(x=seriesform.mean(), color='b', linestyle ='dashed', linewidth = 2)
plt.show()

