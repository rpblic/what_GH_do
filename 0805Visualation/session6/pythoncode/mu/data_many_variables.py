import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import os

os.path.abspath(os.path.curdir)
os.chdir(r'/Users/kangmina')
csv_file=pd.read_csv("HR_comma_sep.csv", dtype={'left':bool,'promotion_last':bool})

making_pair=sns.PairGrid(csv_file, size=1.5)
corr_matrix=making_pair.map_offdiag(plt.scatter, kw=1)
corr_matrix=making_pair.map_diag(plt.hist)
plt.show(corr_matrix)