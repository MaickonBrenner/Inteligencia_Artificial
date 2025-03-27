import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("EMGsDataset.csv", delimiter=',')
data = data.T # Transpõe os dados

c1, c2, c3, c4, c5 = 1, 2, 3, 4, 5 # Classes

plt.figure(1)
plt.title("Visualização de Classes")
plt.scatter(data[data[:,-1]==c1,0],data[data[:,-1]==c1,1], label='Neutro',c='yellow',ec='k')
plt.scatter(data[data[:,-1]==c2,0],data[data[:,-1]==c2,1], label='Sorrindo',c='g',ec='k') # teal
plt.scatter(data[data[:,-1]==c3,0],data[data[:,-1]==c3,1], label='Sobrancelhas Levantadas',c='m',ec='k')
plt.scatter(data[data[:,-1]==c4,0],data[data[:,-1]==c4,1], label='Surpreso',c='b',ec='k')
plt.scatter(data[data[:,-1]==c5,0],data[data[:,-1]==c5,1], label='Rabugento',c='r',ec='k')
plt.xlabel("Corrugador do Supercílio")
plt.ylabel("Zigomático Maior")
plt.legend()

X = np.vstack((
    data[data[:,-1]==c1,:2],
    data[data[:,-1]==c2,:2],
    data[data[:,-1]==c3,:2],
    data[data[:,-1]==c4,:2],
    data[data[:,-1]==c5,:2],
))

Y = np.vstack((
    np.tile(np.array([[1, 0, 0, 0, 0]]),(10000,1)),
    np.tile(np.array([[0, 1, 0, 0, 0]]),(10000,1)),
    np.tile(np.array([[0, 0, 1, 0, 0]]),(10000,1)),
    np.tile(np.array([[0, 0, 0, 1, 0]]),(10000,1)),
    np.tile(np.array([[0, 0, 0, 0, 1]]),(10000,1))
))

N,p = X.shape
C = 5

# Variáveis para os modelos gaussianos bayesianos
X_bayes = data[:p, :]
Y_bayes = np.zeros((C,N))

for i in range(N):
    Y_bayes[:,i] = Y[i, :]

rodadas = 1 #500
particionamento = .8
lambdas = [0, 0.25, 0.5, 0.75, 1]

# Desempenhos
modelo_mqo_tradicional = []
modelo_gaussiano_trad = []
modelo_gaussiano_cov = []
modelo_gaussiano_agre = []
modelo_gaussiano_reg = []
modelo_naive_bayes = []

def estimar_parametro_gaussianos(X_treino, Y_treino, n_classes):
    medias = []
    covariancias = []
    for i in range(n_classes):
        X_classe = X_treino[Y_treino[:,i]==1]
        medias.append(np.mean(X_classe, axis=0))
        covariancias.append(np.cov(X_classe.T))
    return np.array(medias), np.array(covariancias)

def calcular_gaussiana(X, media, cov):
    d = len(media)
    cov += 1e-6 * np.eye(cov.shape[0]) 
    cov_inv = np.linalg.inv(cov)
    normalizador = (2 * np.pi) ** (d / 2) * np.linalg.det(cov) ** 0.5
    diff = X - media
    expoente = -0.5*np.sum(diff@cov_inv*diff, axis=1)
    return np.exp(expoente)/normalizador


for r in range(rodadas):
    # Embaralhar os dados
    idx = np.random.permutation(N)

    # Particionamento
    tamanho_treino = int(N * particionamento)

    # MQO
    X_treino, X_teste = X[idx[:tamanho_treino]], X[idx[tamanho_treino:]]
    Y_treino, Y_teste = Y[idx[:tamanho_treino]], Y[idx[tamanho_treino:]]

    # Estimação dos modelos:
    # Classificação por MQO Tradicional
    X_mqo = np.hstack((np.ones((tamanho_treino,1)),X_treino))

    beta_mqo = np.linalg.pinv(X_mqo.T@X_mqo)@X_mqo.T@Y_treino

    X_teste_com_bias = np.hstack((np.ones((len(X_teste), 1)), X_teste))
    y_predicao_mqo = X_teste_com_bias@beta_mqo
    modelo_mqo_tradicional.append(np.mean(np.argmax(Y_teste, axis=1) == np.argmax(y_predicao_mqo, axis=1)))

    # Classificador Gaussiano Tradicional 
    medias, covs = estimar_parametro_gaussianos(X_treino, Y_treino, 5)
    probs = np.array([calcular_gaussiana(X_teste, medias[i], covs[i]) for i in range(5)]).T
    y_predicao_gaussiano_trad = np.argmax(probs, axis=1)
    modelo_gaussiano_trad.append(np.mean(np.argmax(Y_teste, axis=1) == y_predicao_gaussiano_trad))

    # Classificador Gaussiano Com Covariâncias Iguais
    cov_unica = np.mean(covs, axis=0)
    probs_cov_unica = np.array([calcular_gaussiana(X_teste, medias[i], cov_unica) for i in range(5)]).T
    y_predicao_cov = np.argmax(probs_cov_unica, axis=1)
    modelo_gaussiano_cov.append(np.mean(np.argmax(Y_teste, axis=1) == y_predicao_cov))
    
    # Classificador Gaussiano com Matriz Agregada
    cov_agre = np.cov(X_treino.T)
    probs_agre = np.array([calcular_gaussiana(X_teste, medias[i], cov_unica) for i in range(5)]).T
    y_predicao_agre = np.argmax(probs_agre, axis=1)
    modelo_gaussiano_agre.append(np.mean(np.argmax(Y_teste, axis=1) == y_predicao_agre))

    # Classificador Gaussiano Regularizado (Friedman)
    cov_reg = [(1 - lambdas[0])*covs[i]+lambdas[0] * np.eye(p) for i in range(5)]
    probs_reg = np.array([calcular_gaussiana(X_teste, medias[i], cov_unica) for i in range(5)]).T
    y_predicao_reg = np.argmax(probs_reg, axis=1)
    modelo_gaussiano_reg.append(np.mean(np.argmax(Y_teste, axis=1) == y_predicao_reg))

    # Classificador de Bayes Ingênuo (Naive Bayes)
    # variancias = np.var(X_treino, axis=0)
    # probs_naive = np.array([np.prod(calcular_gaussiana(X_teste, medias[i], np.diag(variancias)), axis=1) for i in range(5)]).T
    # y_predicao_naive = np.argmax(probs_naive, axis=1)
    # modelo_naive_bayes.append(np.mean(np.argmax(Y_teste, axis=1) == y_predicao_naive))

# Gráfico MQO
MQO = {
    'Média': np.mean(modelo_mqo_tradicional), # Média
    'Desvio Padrão': np.std(modelo_mqo_tradicional), # Desvio Padrão
    'Valor Máximo': np.max(modelo_mqo_tradicional), #  Valor máximo
    'Valor Mínimo': np.min(modelo_mqo_tradicional) # Valor mínimo
}
plt.figure(2)
plt.title("Classificação por MQO Tradicional")
plt.bar(MQO.keys(), MQO.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold', 'k'])
plt.xlabel("Medidas")
plt.ylabel("Valor")

# Gráfico Classificador Gaussiano Tradicional 
Gau_Tra = {
    'Média': np.mean(modelo_gaussiano_trad), # Média
    'Desvio Padrão': np.std(modelo_gaussiano_trad), # Desvio Padrão
    'Valor Máximo': np.max(modelo_gaussiano_trad), #  Valor máximo
    'Valor Mínimo': np.min(modelo_gaussiano_trad) # Valor mínimo
}
plt.figure(3)
plt.title("Classificação por MQO Tradicional")
plt.bar(Gau_Tra.keys(), Gau_Tra.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold', 'k'])
plt.xlabel("Medidas")
plt.ylabel("Valor")

# Gráfico Classificador Gaussiano Com Covariâncias Iguais
Gau_Cov = {
    'Média': np.mean(modelo_gaussiano_cov), # Média
    'Desvio Padrão': np.std(modelo_gaussiano_cov), # Desvio Padrão
    'Valor Máximo': np.max(modelo_gaussiano_cov), #  Valor máximo
    'Valor Mínimo': np.min(modelo_gaussiano_cov) # Valor mínimo
}
plt.figure(4)
plt.title("Classificador Gaussiano Com Covariâncias Iguais")
plt.bar(Gau_Cov.keys(), Gau_Cov.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold', 'k'])
plt.xlabel("Medidas")
plt.ylabel("Valor")

# Gráfico Classificador Gaussiano com Matriz Agregada
Gau_Agre = {
    'Média': np.mean(modelo_gaussiano_agre), # Média
    'Desvio Padrão': np.std(modelo_gaussiano_agre), # Desvio Padrão
    'Valor Máximo': np.max(modelo_gaussiano_agre), #  Valor máximo
    'Valor Mínimo': np.min(modelo_gaussiano_agre) # Valor mínimo
}
plt.figure(5)
plt.title("Classificador Gaussiano com Matriz Agregada")
plt.bar(Gau_Agre.keys(), Gau_Agre.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold', 'k'])
plt.xlabel("Medidas")
plt.ylabel("Valor")

# Gráfico Classificador Gaussiano Regularizado (Friedman)
Gau_Reg = {
    'Média': np.mean(modelo_gaussiano_reg), # Média
    'Desvio Padrão': np.std(modelo_gaussiano_reg), # Desvio Padrão
    'Valor Máximo': np.max(modelo_gaussiano_reg), #  Valor máximo
    'Valor Mínimo': np.min(modelo_gaussiano_reg) # Valor mínimo
}
plt.figure(6)
plt.title("Classificador Gaussiano Regularizado (Friedman)")
plt.bar(Gau_Reg.keys(), Gau_Reg.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold', 'k'])
plt.xlabel("Medidas")
plt.ylabel("Valor")

# Gráfico Classificador de Bayes Ingênuo (Naive Bayes)
# Gau_Naive = {
#     'Média': np.mean(modelo_naive_bayes), # Média
#     'Desvio Padrão': np.std(modelo_naive_bayes), # Desvio Padrão
#     'Valor Máximo': np.max(modelo_naive_bayes), #  Valor máximo
#     'Valor Mínimo': np.min(modelo_naive_bayes) # Valor mínimo
# }
# plt.figure(7)
# plt.title("Classificador de Bayes Ingênuo (Naive Bayes)")
# plt.bar(Gau_Naive.keys(), Gau_Naive.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold', 'k'])
# plt.xlabel("Medidas")
# plt.ylabel("Valor")

plt.show()
