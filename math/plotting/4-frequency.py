#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Plot the histogram
plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

# Set the x-axis and y-axis labels
plt.xlabel('Grades')
plt.ylabel('Number of Students')

# Set the title of the plot
plt.title('Project A')

# Show the plot
plt.show()
