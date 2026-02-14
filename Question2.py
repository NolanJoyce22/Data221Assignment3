import pandas as pd
import matplotlib.pyplot as plt

crime_data_frame = pd.read_csv("crime.csv")
# Read the data set using pd
violent_crimes_per_pop_data = crime_data_frame["ViolentCrimesPerPop"]
# Create a new variable with the data column we want to work with

plt.hist(violent_crimes_per_pop_data, bins = 18, edgecolor = "black")
# Create the histogram of the data
plt.title ("Histogram of Violent Crimes Per Population")
# Label the histogram
plt.xlabel("Violent Crimes Per Population")
# Label the x-axis
plt.ylabel("Frequency")
# Label the y-axis
plt.show()
# Display the histogram

plt.boxplot(violent_crimes_per_pop_data)
# Create a boxplot for the data
plt.title("Box Plot of Violent Crimes Per Population")
# Create a title for the box plot
plt.xlabel("Distribution of Violent Crimes Per Population")
# Label the x-axis
plt.ylabel("Values")
# Label the y-axis
plt.show()
# Display the box plot

'''
The distribution in the histogram shows that the data is moderately right skewed, as the majority of the data points 
fall on the left side of the diagram. This supports why the mean is greater than the median.

The box plot shows that the median is a little below the true middle of the box, meaning that more data points appear in
the lower quartile of the box, hence the right skew.

The box plot has quite a high upper fence suggesting that their is some outliers in the upper portion of the data. The
lower fence is a moderate distance from the box, but not as extreme the upper fence, so most likely there isn't any 
extreme outliers there. 

'''
