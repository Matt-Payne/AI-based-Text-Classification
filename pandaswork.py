from __future__ import print_function

import pandas as pd
import numpy  as np
pd.__version__
#The primary data structures in pandas are implemented as two classes:
#DataFrame, which you can imagine as a relational data table, with rows and named columns.
# Series, which is a single column. A DataFrame contains one or more Series and a name for each Series.


#Series
input = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
print(input)

# create some Series, map them to data DataFrame

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

pd.DataFrame({ 'City name': city_names, 'Population': population })

# load file into DataFrame
california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
# describe shows interesting statistics about the datasets
california_housing_dataframe.describe()

# shows first few records
california_housing_dataframe.head()

# create a graph with the housing housing_median_age data
california_housing_dataframe.hist('housing_median_age')

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(type(cities['City name']))
cities['City name']
print(cities['City name'])

print(type(cities['City name'][1]))
cities['City name'][1]

# you can use basic arithmetic to Series
print(population / 100)

#Series can be used as agruments in numpy
np.log(population)

# creates a Series that tells if population is over 1 mill
population.apply(lambda val: val > 1000000)

#modify data frames, addes two Series to an existing DataFrame
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
