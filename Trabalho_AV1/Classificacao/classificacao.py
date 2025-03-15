import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("EMGsDataset.csv", delimiter=',')
data = data.T

c1, c2, c3 = 2,4,5

plt.scatter(data[data[:,-1]==c1,0],data[data[:,-1]==c1,1], label='Sorrindo',c='yellow',ec='k')
plt.scatter(data[data[:,-1]==c2,0],data[data[:,-1]==c2,1], label='Surpreso',c='teal',ec='k')
plt.scatter(data[data[:,-1]==c3,0],data[data[:,-1]==c3,1], label='Rabugento',c='r',ec='k')
plt.xlabel("Corrugador do Supercílio")
plt.ylabel("Zigomático Maior")
plt.legend()
plt.show()
