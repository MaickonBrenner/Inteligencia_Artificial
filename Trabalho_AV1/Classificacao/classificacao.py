import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("EMGsDataset.csv", delimiter=',')
data = data.T

c1, c2, c3, c4, c5 = 1, 2, 3, 4, 5

plt.scatter(data[data[:,-1]==c1,0],data[data[:,-1]==c1,1], label='Neutro',c='yellow',ec='k')
plt.scatter(data[data[:,-1]==c2,0],data[data[:,-1]==c2,1], label='Sorrindo',c='g',ec='k') # teal
plt.scatter(data[data[:,-1]==c3,0],data[data[:,-1]==c3,1], label='Sobrancelhas Levantadas',c='m',ec='k')
plt.scatter(data[data[:,-1]==c4,0],data[data[:,-1]==c4,1], label='Surpreso',c='b',ec='k')
plt.scatter(data[data[:,-1]==c5,0],data[data[:,-1]==c5,1], label='Rabugento',c='r',ec='k')
plt.xlabel("Corrugador do Supercílio")
plt.ylabel("Zigomático Maior")
plt.legend()




plt.show()
