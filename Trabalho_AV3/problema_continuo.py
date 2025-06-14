import numpy as np
import matplotlib.pyplot as plt
from otimizacao import *;

def f1(x1, x2):
    return (x1**2 + x2**2)

def f2(x1,x2):
    return np.exp(-(x1**2 + x2**2)) + 2*np.exp(-((x1-1.7)**2 + (x2-1.7)**2))

def f3(x1,x2):
    return (-20 * np.exp(-0.2*np.sqrt(0.5*(x1**2 + x2**2))) - -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))))

def f4(x1,x2):
    return (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)

def f5(x1,x2):
    return ((x1 * np.cos(x1))/20) + 2 * np.exp(-(x1)**2 - (x2 - 1)**2) + 0.01*x1*x2 

def f6(x1,x2):
    return x1 * np.sin(4 * np.pi * x1) -x2 * np.sin(4 * np.pi * x2 + np.pi)

def f7(x1,x2):
    return -np.sin(x1)*np.sin(x1**2/np.pi)**2*10 - np.sin(x2)*np.sin(x2**2/np.pi)**2*10

def f8(x1,x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

restricoesf1 = np.array([
    [-100,100],
    [-100,100]
])

restricoesf2 = np.array([
    [-2,4],
    [-2,5]
])

restricoesf3 = np.array([
    [-8,8],
    [-8,8]
])

restricoesf4 = np.array([
    [-5.12,5.12],
    [-5.12,5.12]
])

restricoesf5 = np.array([
    [-10,10],
    [-10,10]
])

restricoesf6 = np.array([
    [-1,3],
    [-1,3]
])

restricoesf7 = np.array([
    [0, np.pi],
    [0, np.pi]
])

restricoesf8 = np.array([
    [-200,200],
    [-200,200],
])

# x1 = np.linspace (-100, 100, 1000)
# X1, X2 = np.meshgrid(x1, x1)
# Y = f(X1, X2)
# x1_cand , x2_cand = 50, 50
# f_cand = f ( x1_cand , x2_cand )
# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.plot_surface (X1, X2, Y, rstride=10 , cstride=10 , alpha=0.6, cmap="jet")
# ax.scatter(x1_cand, x2_cand, f_cand , marker = "x" , s=90, linewidth=3, color="red")
# ax.set_xlabel ("x")
# ax.set_ylabel ("y")
# ax.set_zlabel ("z")
# ax.set_title("f(x1, x2)")
# plt.tight_layout()
# plt.show()

# globa = GlobalRandomSearch(1000,f7,restricoesf7)
# globa.search()


hill = HillClimbing(1000,20,.5,f8,restricoesf8)
hill.search()

# problema = int(input("Digite o valor do problema: "))

# if problema == 1:
    

#     # local = LocalRandomSearch(1000,.5,f2,restricoesf2)
#     # local.search()

#     hill = HillClimbing(1000,20,.5,f3,restricoesf3)
#     hill.search()
    
# elif problema == 2:

#     print()
# elif problema == 3:
#     print()
# elif problema == 4:
#     print()
# elif problema == 5:
#     print()