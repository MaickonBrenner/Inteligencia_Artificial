import numpy as np
import matplotlib.pyplot as plt

# Carregar dados
data = np.loadtxt("atividade_enzimatica.csv", delimiter=',')

X = data[:, :2]  # Temperatura e pH
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

for r in range(rodadas):
    # Embaralhar os dados
    idx = np.random.permutation(N)
    Xr = X[idx, :]
    yr = y[idx, :]

    # Particionar os dados (80% treino, 20% teste)
    split = int(N * particionamento)
    X_treino, y_treino = Xr[:split, :], yr[:split, :]
    X_teste, y_teste = Xr[split:, :], yr[split:, :]

    # Modelo da Média dos valores observáveis
    beta_media = np.mean(y_treino)  # Apenas a média, sem matriz
    y_pred_media = np.full(y_teste.shape, beta_media)  # Preenche com o valor médio
    modelo_media.append(np.sum((y_teste - y_pred_media) ** 2))

    # MQO Tradicional
    # Adicionar coluna de 1s para o intercepto
    X_treino = np.hstack((np.ones((X_treino.shape[0], 1)), X_treino))
    X_teste = np.hstack((np.ones((X_teste.shape[0], 1)), X_teste))
    beta_mqo = np.linalg.pinv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
    y_pred_mqo = X_teste @ beta_mqo
    modelo_mqo.append(np.sum((y_teste - y_pred_mqo) ** 2))

# Função para calcular métricas
def calcular_metricas(modelo):
    return np.mean(modelo), np.std(modelo), np.max(modelo), np.min(modelo)

# Exibir os resultados no console
print("\nResultados:")
print(f"{'Modelo':<20}{'Média RSS':<15}{'Desvio-Padrão':<15}{'Maior Valor':<15}{'Menor Valor'}")
print("=" * 70)

metricas = [
    ("Média", modelo_media),
    ("MQO", modelo_mqo),
]

for nome, modelo in metricas:
    media, std, maior, menor = calcular_metricas(modelo)
    print(f"{nome:<20}{media:<15.4f}{std:<15.4f}{maior:<15.4f}{menor:.4f}")

# Boxplot para comparação dos modelos
plt.figure(figsize=(10, 5))
plt.boxplot([modelo_media, modelo_mqo], labels=["Média", "MQO"])
plt.ylabel("RSS")
plt.title("Comparação dos Modelos")
plt.show()
