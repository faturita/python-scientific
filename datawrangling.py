'''
Pandas.  Data Wrangling in python.

if you ever work as a fancy Data Scientists, it is very likely that 80% of your time will be spent doing data wrangling (I prefer the term Data Cooking).
This will continue until you finally achieve a managerial postion and stop doing useful work, all of a sudden.


Input data   -->   Data Cooking ---> Visualization
                        |
                    Intermediate Files


Software development, in any form, is a human endeavor: it requires to articulate people working together in teams.
TDSP: Team Data Science Process

Input Data:
⏺   Data Sources

Data Cooking:
⏺   Dealing with missing data.
⏺   ETL: extract transform and load.  DTS: Data tranformation services
⏺   

Visualization
⏺   fdsfsdadfa

This script contains several snippets, separtaed by '# %%' which is a special marker that allows Visual Studio Code
to treat a python script as a jupyter notebook.  This is quite good, because it allows to use both systems at the same
time.

References:
- Wes McKinney, Python for Data Analysis, 2017
- Harrison, Learning the Pandas Library, 2016

'''

# %% 
print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

print('Hello Python Scientific World')

online = False
if (online == True):
    url = requests.get('https://drive.google.com/file/d/117pqjcY15qMGY0HlFaEz195_7uuq6LBv/view?usp=sharing')
    csv_raw = StringIO(url.text)
    signals = pd.read_csv(csv_raw, delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
else:
    signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])

print('Information:')
print(signals.head())

print('Filter records:')
print(signals[signals.counter > 45])

# %%
print('Moving to numpy !')
data = signals.values

print('Now in "data", you have a tensor.')
print (data)

print('Shape %2d,%2d:' % (signals.shape))

print('From here you can start working around the data structure which has a mathematical purpose.')


# %% 
print('You can go the other way around and convert a numpy array into a dataframe.')

databack = pd.DataFrame(data, columns=['ts', 'ct', 'e','att','med','blk'])

print ('Shape %2d,%2d:' % databack.shape)

# %%
print('Visualizations')

import seaborn as sns
sns.set(style="darkgrid")
sns.lineplot(x="timestamp", y="eeg", hue="attention", data=signals)
import matplotlib.pyplot as plt
plt.show()


# %%

print('The whole picture, working in Data Science Project in corporate environments')
# https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview

# %%
print('----------------------------------------------------------------------------------------------------------------------------------')
print('Pandas is great to read data from several differentes sources.')

jsonlike = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}

mydataframe = pd.DataFrame(jsonlike)
print(mydataframe)

# %%

print(' read_xxxx methods load data in several formats.  This like Excel import, so there are hundreds of parameters.')
signals = pd.read_csv('data/blinking.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
# //skip_rows, index_cols, sep='\s+', nrows

signals.head()

print('read_sas, read_sql, read_pickle, read_json, read_excel, ...')


dat = pd.read_csv('data/laliga.csv',delimiter=';')

# %%
print('Data can finally be exported to an output file.')
dat.to_csv('data/out.csv')

# %% 
print('JSON is a widespread format very used in web environments.')
obj = """
    {"name": "Wes",
     "places_lived": ["United States", "Spain", "Germany"],
     "pet": null,
     "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
                  {"name": "Katie", "age": 38,
                   "pets": ["Sixes", "Stache", "Cisco"]}]
} """

import json

results = json.loads(obj)

results

siblings = pd.DataFrame(results['siblings'], columns=['name', 'age'])

print(siblings)

json_string = json.dumps(results)


# %%

data = pd.read_json('data/sample2.json')

print('It is possible to put it back to json string.')
print(data.to_json())


# %%

tables = pd.read_html('http://monostuff.logdown.com/')

journals = tables[0]

# %%
print('Python uses pickles to serialize structures, and this can be read from pandas.')

calories = {"day1": 420, "day2": 380, "day3": 390}

cal_frame = pd.Series(calories)

cal_frame.to_pickle('data/frame.pickle')


new_frame = pd.read_pickle('data/frame.pickle')


# %% 
print('Reading some json data out of a API.')
from io import StringIO
import requests

url = requests.get('https://api.github.com/repos/faturita/python-scientific/commits')
js = StringIO(url.text)
sg = pd.read_json(js)
sg.head()


# %%
print('-------------------------------------------------------------------------------------')
print('Panda detects missing data by checking sentinel values like "NaN", "NULL", "NA".')
results = pd.read_csv('data/sample1.csv')

print('You can get an indicative matrix which returns true on the cell where the data is missing')
results.isnull()

results.notnull()

print('It is possible to specify different sentinel values for missing data.')
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}

print('For each column,  the sentinel value.')
results = pd.read_csv('data/sample1.csv', na_values = sentinels)
print ( results )


# %% 
print('Handling missing data.')

from numpy import nan as NA
data = pd.Series([1, NA, 3.5, NA, 7])

data.dropna()

print('The same as')

data[data.notnull()]

# %% 
print('On 2D')
data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
data.dropna(how='all', axis = 1)        # 0 is row, 1 is col


# %%
print('Now fillna can be used in the same way to fill in the data.')

data.fillna(0)

data.fillna({0: 0.5, 2:0})

print('An interpolation can be performed to fill in the missing values.')
data.fillna(method='ffill')

data[np.abs(data) > 1.5] = np.sign(data) * 10

data.describe()

# %% 
print('None which is the python Null marker, can also be considered as NA sentinel.')
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])

string_data[0] = None

string_data


# %%
print('Removing duplicates')
string_data = pd.Series(['Newark', 'Manchester', 'Halifax', 'Manchester'])

string_data.drop_duplicates()

string_data.replace({'Newark': np.nan, 'Halifax': 'Ottawa'})

# %%
print('----------------------------------------------------------------------------------------------------------------------------------')
print('Pandas Series: data structure to handling sequential data.')

import pandas as pd

a = [1, 7, 2]

series = pd.Series(a)

print(series)

print('By default, indexes are created with range on the number of elements.')
series = pd.Series(a, index = ["x", "y", "z"])

print(series)
print(series.values)
print(series.index)


# %%
print('String indexes can be used on series.')
a = [1, 7, 2]
series = pd.Series(a, index = ["x", "y", "z"])

print(series['x'])

series[series > 1]

series * 2

np.exp(series)

'x' in series 

'r' in series 
# %%
print('Index are automatically used to operate on the data.')
a = [1, 7, 2]
series1 = pd.Series(a, index = ["x", "y", "z"])
b = [-1, -8, 3]
series2 = pd.Series(a, index = ["y", "z", "r"])

series3 = series1 + series2 


# %%
print('Dataframes are dynamics structures')
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
            'year': [2000, 2001, 2002, 2001, 2002, 2003],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)


frame.state 

frame.index = ['one','two','three','four','five','six']

frame['pop'] = 9.0              # References the column 'pop'
frame.loc['three']              # References the row indexed 'three'

# %%
print('Assigning several values at the same time.')
val = pd.Series([11, 12, 13], index=['two', 'four', 'five'])

frame['pop'] = val              # This means that pop is REPLACED by val.


frame['Casinos'] = frame.state == 'Nevada'

del frame['Casinos']            # Removes the column 'Casinos'

frame.columns.name = 'Variables'
frame.index.name = 'Locations'

# %%
print('Reindexing: mapping the values with new indexes.')
a = [1, 7, 2]
series1 = pd.Series(a, index = ["x", "y", "z"])

series2 = series1.reindex(["x","z","e"])

# %%
