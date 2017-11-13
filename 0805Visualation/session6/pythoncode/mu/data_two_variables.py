import os
import pandas as pd
import matplotlib.pyplot as plt

os.path.abspath(os.path.curdir)
os.chdir(r'/Users/kangmina')
csv_file=pd.read_csv("HR_comma_sep.csv", dtype={'left':bool,'promotion_last':bool})
satisfaction_level=csv_file['satisfaction_level']
last_evaluation=csv_file['last_evaluation']

csv_file.describe()
satisfaction_level=csv_file['satisfaction_level']
last_evaluation=csv_file['last_evaluation']
average_montly_hours=csv_file['average_montly_hours']

plt.style.use('ggplot')
plt.scatter(average_montly_hours,satisfaction_level,marker='.',color='red',label='satisfaction_level', s=3)
plt.scatter(average_montly_hours,last_evaluation,marker='.',color='gray',label='last_evaluation', s=3)
plt.xlabel('average_montly_hours')
plt.ylabel('satisfaction_level & last_evaluation')
plt.title('Joint Distribution')
plt.show()

print(satisfaction_level.corr(average_montly_hours))
print(last_evaluation.corr(average_montly_hours))