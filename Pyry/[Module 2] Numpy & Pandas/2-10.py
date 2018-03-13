import pandas as pd

data = pd.read_excel('animals.xlsx')
print(data)

animals = data.loc[:, ['name', 'hair', 'feathers']]
print(animals)