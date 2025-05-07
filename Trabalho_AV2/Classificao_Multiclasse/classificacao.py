import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('coluna_vertebral.csv',delimiter=',')

X = data[:,:6].astype(float)

print("Formato de X:", X.shape) 


Y = np.vstack((
    np.tile(np.array([[1, -1, -1]]),(10000,1)), # Normal
    np.tile(np.array([[-1, 1, -1]]),(10000,1)), # Hérnia de Disco
    np.tile(np.array([[-1, -1, 1]]),(10000,1)), # Espondilolistese
))