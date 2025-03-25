import numpy as np
import matplotlib.pyplot as plt

# Carregar dados
data = np.loadtxt("atividade_enzimatica.csv", delimiter=',')

X = data[:, :2]  # Temperatura e pH (variáveis independentes)
y = data[:, 2:]  # Nível de atividade enzimática 
N, p = X.shape

# Visualização inicial
plt.figure(0)
ax = plt.subplot(projection='3d')
ax.scatter(X[:, 0], X[:, 1], y[:, 0], c='purple', edgecolor='k')
ax.set_xlabel("Temperatura")
ax.set_ylabel("pH da Solução")
ax.set_zlabel("Nível de Atividade Enzimática")
ax.set_title("Análise dos Níveis de Atividade Enzimática")
plt.show()

# Simulações de Monte Carlo
rodadas = 500
particionamento = 0.8

modelo_media = []
modelo_mqo = []
modelo_tikhonov = {l: [] for l in [0, 0.25, 0.5, 0.75, 1]}

for r in range(rodadas):
    idx = np.random.permutation(N)
    Xr, yr = X[idx, :], y[idx, :] 

    split = int(N * particionamento)
    X_treino, y_treino = Xr[:split, :], yr[:split, :]
    X_teste, y_teste = Xr[split:, :], yr[split:, :]


    # Modelo da Média dos valores observáveis
    beta_media = np.mean(y_treino, axis=0)  # Mantém formato correto (1,1)
    y_pred_media = np.full(y_teste.shape, beta_media)  # Mantém y_pred_media como matriz (N,1)

    # Cálculo do RSS para o modelo da Média
    modelo_media.append(np.sum((y_teste - y_pred_media) ** 2))

    # MQO Tradicional
    # Adicionar coluna de 1s para o intercepto
    X_treino = np.hstack((np.ones((X_treino.shape[0], 1)), X_treino))
    X_teste = np.hstack((np.ones((X_teste.shape[0], 1)), X_teste))
    beta_mqo = np.linalg.pinv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
    y_pred_mqo = X_teste @ beta_mqo

    # Cálculo do RSS para o MQO Tradicional
    modelo_mqo.append(np.sum((y_teste - y_pred_mqo) ** 2))

    # MQO Regularizado (Tikhonov)
    I = np.eye(p + 1)
    I[0, 0] = 0  # Não penalizar o intercepto
    for l in modelo_tikhonov.keys():
        beta_tikhonov = np.linalg.pinv(X_treino.T @ X_treino + l * I) @ X_treino.T @ y_treino
        y_pred_tikhonov = X_teste @ beta_tikhonov

        # Cálculo do RSS para o MQO Regularizado (Tikhonov)
        modelo_tikhonov[l].append(np.sum((y_teste - y_pred_tikhonov) ** 2))

# Exibição dos resultados
print("\n Comparação dos Modelos de Regressão\n")
print(f"{'Modelo':<25}{'Média RSS':<15}{'Desvio-Padrão':<15}{'Maior Valor':<15}{'Menor Valor'}")
print("=" * 80)

metricas = [("Média", modelo_media), ("MQO", modelo_mqo)] + [
    (f"Tikhonov λ={l}", modelo_tikhonov[l]) for l in modelo_tikhonov.keys()
]

for nome, modelo in metricas:
    media, std, maior, menor = np.mean(modelo), np.std(modelo), np.max(modelo), np.min(modelo)
    print(f"{nome:<25}{media:<15.4f}{std:<15.4f}{maior:<15.4f}{menor:.4f}")

# Boxplot para comparação dos modelos
plt.figure(figsize=(10, 5))
plt.boxplot([modelo_media, modelo_mqo] + list(modelo_tikhonov.values()), 
            labels=["Média", "MQO"] + [f"Tikhonov λ={l}" for l in modelo_tikhonov.keys()])
plt.ylabel("RSS")
plt.title("Comparação dos Modelos")
plt.show()
