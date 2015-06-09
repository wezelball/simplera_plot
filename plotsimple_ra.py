#/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import csv
import time
import sys, os, string
from datetime import datetime
from pylab import figure
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import stats

# The final array of datetime in UTC
utcTime = np.array([])
# The final array of total power as a float
totalPower = np.array([])

# Set the year, month, and day to today
utcyear = time.strftime("%Y")
utcmonth = time.strftime("%m")
utcday = time.strftime("%d")

# These are lists that are converted into the arrays above
# It's done that way cause its much easier to append to a list
utime = []
tpower = []
utime_num = []

# My zenity script passes all arguments as one
arguments = sys.argv[1]
# The split method creates a list of 3 elements
# inputfile, startTrim, endTrim
inputfile = arguments.split()[0] 

# make sure trim values exist
#if arguments.split()[1] == "":
#    arguments.split()[1] = 0
#if arguments.split()[2] == "":
#    arguments.split()[2] = 0

startTrim = int(arguments.split()[1])
endTrim = int(arguments.split()[2])

# Open the file
f = open(inputfile, 'rt')

try:
    # Read the file as a CSV, comma-delimited and ' quoted
    reader = csv.reader(f, delimiter=',', quotechar = "'")
    for row in reader:
        # UTC hour is 1st column
        utchour = row[0]
        # Then UTC minute
        utcminute = row[1]
        # Then UTC second
        utcsecond = row[2]
        # Build the UTC string properly colon separated and spaces removed
        utcString = string.strip(utchour + ":" + utcminute + ":" + utcsecond)
        # Build the total power string with spaces removed
        tString = string.strip(row[6])
        # Universal time as a list of datetime.datetime
        utime.append(datetime(int(utcyear), int(utcmonth), int(utcday),int(utchour), int(utcminute), int(utcsecond)))
        # tpower as a list of floats
        tpower.append(float(tString))
        
finally:
    # Close file when done
    f.close()

# These are np arrays created from the lists defined    
# This is what is plotted on x axis
# These are the full-length arrays, before trimming
utcTimeFull = np.array(utime)
# This is converted to numbers that we can perform statistics on
utcTimeNumFull = np.array(dates.date2num(utime))
totalPowerFull = np.array(tpower)

# **************************
# This is where we need to trim off the ends of the arrays to get rid of "crap"
utcTime = utcTimeFull[startTrim:len(utcTimeFull)-endTrim]
utcTimeNum = utcTimeNumFull[startTrim:len(utcTimeNumFull)-endTrim]
totalPower = totalPowerFull[startTrim:len(totalPowerFull)-endTrim]

# Determine linear regression statistics
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(utcTimeNum, totalPower)
# Calculate some additional outputs
# predict_y is the actual linear regression
predict_y = intercept + slope * utcTimeNum
pred_error = totalPower - predict_y
degrees_of_freedom = len(utcTime) - 2
residual_std_error = np.sqrt(np.sum(pred_error**2) / degrees_of_freedom)

#------------------------------------------------
# Create a FIR filter and apply it to x.
#------------------------------------------------

# Sample rate is one sample every 5 secs
sample_rate = 100

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 5 Hz transition width.
width = 5.0/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 60.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)

# The cutoff frequency of the filter.
# These values are not numerically accurate, but they produce good 
# results in eliminating noise - 0.005 is a good starting point
#cutoff_hz = 0.01
cutoff_hz = 0.005

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

# Use lfilter to filter x with the FIR filter.
filteredPower = lfilter(taps, 1.0, totalPower)

# Subtract the linear regression from the filtered data
adjFilteredPower = filteredPower - predict_y

# This is the raw data
figure(1)
plt.title(inputfile)
plt.suptitle("Raw Input Data")
plt.ylabel('Total Power')
plt.xlabel('Time, UTC')

# Plot figure 1, the raw data
plt.plot(utcTime, totalPower)

# This is the filtered data, with the regression
figure(2)
plt.title(inputfile)
plt.suptitle("Filtered Data, with Regression")
plt.ylabel('Total Power')
plt.xlabel('Time, UTC')

# Plot figure 2, the filtered data
plt.plot(utcTime, filteredPower)
plt.plot(utcTime, predict_y)

# This is temperature adjusted data
figure(3)
plt.title(inputfile)
plt.suptitle("Filtered and Temperature Compensated")
plt.ylabel('Total Power')
plt.xlabel('Time, UTC')

# Plot figure 1, the raw data
plt.plot(utcTime, adjFilteredPower)

# Show the plots
plt.show()