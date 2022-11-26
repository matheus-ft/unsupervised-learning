# %%
from movie import (
    compute_cost,
    normalize_ratings,
    gradientDescent,
)
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.io as sio

# %%
data = sio.loadmat("data/dado2.mat")
Y = data["Y"]
R = data["R"]

# %%
print(
    f"Average rating for movie 1 (Toy Story): {np.sum(Y[0, :]*R[0, :]) / np.sum(R[0, :]):.2f}/5"
)

# %%
plt.figure(figsize=(8, 16))
plt.imshow(Y)
plt.xlabel("Users")
plt.ylabel("Movies")

# %%
n_movies, n_users = Y.shape
n_features = 100
X = npr.randn(n_movies, n_features)
Theta = npr.randn(n_users, n_features)
params = np.append(X.flatten(), Theta.flatten())

# %%
J, grad = compute_cost(X, Theta, Y, R)
print("Custo sobre os parâmetros carregados: ", J)

# %%
movieList = open("data/dado3.txt", "r").read().split("\n")[:-1]
movieList

# %%
Ynorm, Ymean = normalize_ratings(Y, R)

# %%
paramsFinal, J_history = gradientDescent(
    params, Ynorm, R, n_users, n_movies, n_features, 0.001, 20, 0
)
plt.plot(J_history)
plt.xlabel("Iterações")
plt.ylabel("$J(\Theta)$")
plt.title("Função de Custo usando Gradiente Descendente")

# %%
X = paramsFinal[: n_movies * n_features].reshape(n_movies, n_features)
Theta = paramsFinal[n_movies * n_features :].reshape(n_users, n_features)

# %%
p = X @ Theta.T
my_predictions = p[:, 0][:, np.newaxis] + Ymean

# %%
import pandas as pd

df = pd.DataFrame(np.hstack((my_predictions, np.array(movieList)[:, np.newaxis])))
df.sort_values(by=[0], ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# %%
print("Melhores recomendações para você:\n")
for i in range(10):
    print("Nota predita", round(float(df[0][i]), 1), " para o índice", df[1][i])
