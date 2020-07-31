from matplotlib import pyplot as plt
import numpy as np
from math import sqrt

datafile = open("../dataCollection/coords.txt", "r")
rows = datafile.read().split('\n')

x_data = []
y_data = []
for row in rows:
    if(len(row.split(",")) == 9):
        rowdata = [float(elem) for elem in row.split(',')]
        x_data.append(rowdata[:-1])
        y_data.append(rowdata[-1])

areas = []
toplenghts = []
rightlenghts = []
bottomlenghts = []
leftlenghts = []
aspectratios = []
avgwidths = []
avgheights = []

outfile = open("sidelengths.txt", "a")

for idx, row in enumerate(x_data):
	area = ((row[0]*row[3] - row[1]*row[2]) +
			(row[2]*row[5] - row[3]*row[4]) +
			(row[4]*row[7] - row[5]*row[6]) +
			(row[6]*row[1] - row[7]*row[0]))/2
	toplenght=(sqrt((row[2]-row[0])**2 + (row[3]-row[1])**2))
	rightlenght=(sqrt((row[4]-row[2])**2 + (row[5]-row[3])**2))
	bottomlenght=(sqrt((row[6]-row[4])**2 + (row[7]-row[5])**2))
	leftlenght=(sqrt((row[0]-row[6])**2 + (row[1]-row[7])**2))
	outfile.write("{},{},{},{},{}\n".format(toplenght, rightlenght, bottomlenght, leftlenght, y_data[idx]))
	
	avgwidth = .5*(toplenght+bottomlenght)
	avgheight = .5*(leftlenght+rightlenght)
	asepctratio = (avgwidth)/(avgheight)

	toplenghts.append(toplenght)
	rightlenghts.append(rightlenght)
	bottomlenghts.append(bottomlenght)
	leftlenghts.append(leftlenght)
	avgwidths.append(avgwidth)
	avgheights.append(avgheight)
	aspectratios.append(asepctratio)
	
	areas.append(area)
	
areas = np.asarray(areas)
y_data = np.asarray(y_data)

plt.subplot(1,3,1)
plt.scatter(y_data, avgwidths)

plt.subplot(1,3,2)
plt.scatter(y_data, toplenghts)

plt.subplot(1,3,3)
plt.scatter(y_data, bottomlenghts)
'''plt.subplot(2,3,1)
plt.scatter(y_data, areas)



plt.subplot(2,3,4)
plt.scatter(y_data, avgheights)

plt.subplot(2,3,2)
plt.scatter(y_data, toplenghts)

plt.subplot(2,3,3)
plt.scatter(y_data, rightlenghts)

plt.subplot(2,3,4)
plt.scatter(y_data, bottomlenghts)

plt.subplot(2,3,5)
plt.scatter(y_data, leftlenghts)'''

plt.show()
 	 
