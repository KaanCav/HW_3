

import matplotlib.pyplot as plt
import numpy as np
def HiFilter(f,R,C):                     # Define the Filter function that corresponds to the low pass RC filter.
    omega = 2*np.pi*f               # defining omega to be used in the next line to calculate Vout
    Vin = 1
    vout= Vin*((1j*omega*R*C)/(1j*omega*R*C + 1)) # formula for high pass filter (see attached picture for derivation)
    return(vout)

t1 = np.loadtxt('Bode_Plot.txt',usecols=(0))
V1 = np.loadtxt('Bode_Plot.txt',usecols=(1))
V2 = np.loadtxt('Bode_Plot.txt',usecols=(2))

Vc1 = np.loadtxt('Linear Attenuation Plot High Pass Filter.csv', usecols=(1))
Vc2 = np.loadtxt('Linear Attenuation Plot High Pass Filter.csv', usecols=(2))

f = np.linspace(10,10000,1000) # this is essentially the wave generator going up in frequency in equal intervals
R=1000.  # 1kOhm
C=100.e-9  # 1nF
vout_c = HiFilter(f,R,C) #This calls the function and then assigns it a variable
F_cut = 1./(2.*np.pi*R*C)  # formula for the cut off frequency
print("========================================")
print("Filter cut off frequency is: {:7.2f} Hz".format(F_cut))      #This is the same as the
print("========================================")
plt.figure(figsize=(15,9)) # Making the figure 15 by 9 inches

plt.subplot(2,1,1)         # First subplot in the figure with (2 rows, 1 column, 1st subplot)

plt.plot(f,np.abs(vout_c),label='Theoretical Output') # plotting the amplitude which is the absolute value of the hifilter function

plt.plot([F_cut,F_cut],[0,1],color="red",label="Cut-off frequency")  # plot a line for the filter cut frequency
plt.plot(t1,Vc1,color='orange',label='Experimental Input')
plt.plot(t1,Vc2,color='purple',label='Experimental Output')
plt.legend(prop={'size': 15})     #labeling the filter cut frequency and then sizing the legend so it looks good
plt.title("Magnitude plot")
plt.xlabel("F [Hz]")
plt.ylabel("V [Volt]")
plt.xscale("log")           # Set x to a log scale
plt.grid(True)

plt.subplot(2,1,2)

plt.plot(f,np.angle(vout_c), label='Theoretical Output') # Plot the amplitude, the absolute of the complex number
plt.plot(t1, (V2*(np.pi/180)),color= 'purple', label='Experimental Input')
plt.title("Phase plot")
plt.xlabel("F [Hz]")
plt.ylabel("Phase Angle [Rad]")
plt.xscale("log")
plt.legend(prop={'size': 15})
plt.grid(True)
plt.tight_layout()          # Automatically adjust spacing between the 2 plots so they do not overlap
plt.show()