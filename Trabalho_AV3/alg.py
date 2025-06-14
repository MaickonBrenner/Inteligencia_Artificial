import numpy as np
import matplotlib.pyplot as plt

def roleta(C,probs):
    i = 0
    s = probs[i]
    r = np.random.uniform()
    while s < r:
        i+=1
        s+= probs[i]
    return C[i,:]


def psi(x,y):
    return np.abs(x*y*np.sin(y*np.pi/4))

def phi(x,inf,sup):
    s = 0
    for i in range(len(x)):
        s += x[len(x)-i-1]*2**i
    return inf + (sup-inf)/(2**len(x)-1)*s

N = 20
nd = 8
p = 2
P = np.random.uniform(low=0,high=2,size=(N,p*nd)).astype(int) #representação canônica

# x,y = phi(P[2,:nd],-1,15),phi(P[2,nd:],-1,15)

x1 = P[17,:]
x2 = P[11,:]
f1 = np.copy(x1)
f2 = np.copy(x2)
m = np.zeros(len(x1))
idx = np.random.randint(low=1,high=len(m))

m[idx:] = 1
f1[m[:]==1] = x2[m[:]==1]
f2[m[:]==1] = x1[m[:]==1]

individuo = np.split(P[2,:],p)
decodificado = [phi(i,-1,15) for i in individuo]
aptidao = psi(*decodificado)

P = np.random.uniform(low=-3,high=20,size=(N,p)) #representação não canônica (contínuo)
P = np.random.uniform(low = 0, high=8, size=(N,8)).astype(int) #discreto
P = np.array([np.random.permutation(p) for i in range(N)]) #discreto - combinatória

C = np.array([
    [1,1,0,0],
    [1,0,0,0],
    [0,1,1,0],
    [0,0,0,1],
])

aptidoes = []
for i in range(C.shape[0]):
    aptidoes.append(psi(phi(C[i,:],0,20), phi(C[i, 2:], 0, 20)))

total = np.sum(aptidoes)
probabilidades = []

for i in range(C.shape[0]):
    probabilidades.append(aptidoes[i]/total)


S = np.empty((0,4))
for i in range(C.shape[0]):
    S = np.concat((
        S,
        roleta(C,probabilidades).reshape(1,4)
    ))

plt.pie(probabilidades,labels=probabilidades)
plt.show()