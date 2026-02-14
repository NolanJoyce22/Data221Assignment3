import pandas as pd
import numpy as np


crime_data_frame = pd.read_csv("crime.csv")
# Read the data set using pd
violent_crimes_per_pop_data = crime_data_frame["ViolentCrimesPerPop"]
# Create a new variable with the data column we want to work with
mean_violent_crimes_per_pop_data = violent_crimes_per_pop_data.mean()
# Calculate the mean of the violent crimes per population
median_violent_crimes_per_pop_data  = violent_crimes_per_pop_data.median()
# Calculate the median of the violent crimes per population
standard_deviation_violent_crimes_per_pop_data = np.std(violent_crimes_per_pop_data)
# Calculate the standard deviation of the violent crimes per population
maximum_value_in_crimes_per_pop = violent_crimes_per_pop_data.max
# Calculate the maximum value of the violent crimes per population
minimum_value_in_crimes_per_pop = violent_crimes_per_pop_data.min
# Calculate the minimum value of the violent crimes per population

print(mean_violent_crimes_per_pop_data)
print(median_violent_crimes_per_pop_data)

'''
Due to the mean being greater than the median the distribution is slightly right skewed. The difference between the
two values is relatively small but their is a slight skew in the data.

If extreme values occur in a data set the mean will be more affected than the median. This is because the mean is
the sum of all values divided by the number of values, meaning an extreme value can drastically
alter the mean. Where as the median is the true measure of center, meaning it is the point that most closely represents
the middle, so their is no calculations involved in finding the median. Due to this it isn't affected by extreme values.

'''




