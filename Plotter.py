import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import seaborn as sns
from matplotlib import ticker
from scipy import stats

# Data for areas
data_rates = [1, 5, 10]  # Common data rates for both DAC
dac_power = [0.12, 26, 30]  # Power consumption for DAC
adc_power = [2.55, 11, 30]  # Power consumption for ADC
dac_area = [0.00007, 0.06, 0.06]  # Area for DAC
adc_area = [2, 21, 103]  # Area for ADC

# Create the figure and the first axis (for power)
fig, ax1 = plt.subplots(figsize=(10,5))

# Plot the power consumption data on the primary y-axis
ax1.plot(data_rates, dac_power, label='DAC Power (mW)', marker='o', color='blue')
ax1.plot(data_rates, adc_power, label='ADC Power (mW)', marker='s', color='red')

# Set labels for the primary y-axis
ax1.set_xlabel('Data Rate (GS/s)')
ax1.set_ylabel('Power (mW)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')

# Create the second y-axis (for area)
ax2 = ax1.twinx()
ax2.plot(data_rates, dac_area, label='DAC Area (mm²)', marker='^', linestyle='--', color='green')
ax2.plot(data_rates, adc_area, label='ADC Area (mm²)', marker='v', linestyle='--', color='purple')

# Set labels for the secondary y-axis
ax2.set_ylabel('Area (mm²)', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

# Add a title and a grid
plt.title('Power Consumption and Area of DAC and ADC at Different Data Rates')
plt.grid(True)

# Set x-axis ticks to match the data rates
ax1.set_xticks(data_rates)

# Tight layout to ensure everything fits without overlapping
fig.tight_layout()

# Save the plot as a file
file_path_with_area = 'power_area_consumption_plot.png'
plt.savefig(file_path_with_area)

plt.show()
