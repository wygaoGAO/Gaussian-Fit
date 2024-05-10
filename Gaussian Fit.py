#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Example data setup (x_data and y_data should be your actual dataset)
x_data = np.linspace(6834600000, 6834700000, 500)  # Replace with your frequency data
y_data = np.sin(2 * np.pi * 0.0000001 * (x_data - 6834650000)) + np.random.normal(0, 0.1, x_data.size)  # Replace with your normalized probability data

# Gaussian function definition
def improved_gaussian(x, A, mu, sigma, B):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2)) + B

# Find peaks in the data
peaks, properties = find_peaks(y_data, height=np.max(y_data) * 0.5, distance=20)

# Choose the peak closest to the center frequency (example: 6834684228.62 Hz)
target_peak = 6834684228.62
closest_peak = min(peaks, key=lambda peak: abs(x_data[peak] - target_peak))
if abs(x_data[closest_peak] - target_peak) > 100000:  # Tolerance for peak proximity
    print("No peak is close enough to the target frequency.")
    closest_peak = None

# Perform Gaussian fitting around the closest peak
if closest_peak is not None:
    local_range = 50  # Adjust as necessary
    indices = slice(max(0, closest_peak - local_range), min(len(x_data), closest_peak + local_range))
    popt, pcov = curve_fit(improved_gaussian, x_data[indices], y_data[indices], p0=[properties["peak_heights"][0], x_data[closest_peak], 10000, min(y_data[indices])])

    # Plot the fitting and the original data
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'o', label='Original Data')
    x_fit = np.linspace(x_data[indices.start], x_data[indices.stop], 300)
    plt.plot(x_fit, improved_gaussian(x_fit, *popt), 'r-', label='Gaussian Fit')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized Probability')
    plt.title('Gaussian Fit for Central Ramsey Fringe')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Could not find a suitable peak to fit.")


# In[ ]:




