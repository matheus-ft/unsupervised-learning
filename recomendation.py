# %%
from movie import compute_cost, fit, predict, init_random_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

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
plt.ylabel("Movies")
plt.xlabel("Users")

# %%
n_movies, n_users = Y.shape
n_features = 100
X = init_random_matrix(n_movies, n_features)
Theta = init_random_matrix(n_users, n_features)
params = np.append(X.flatten(), Theta.flatten())

# %%
J = compute_cost(X, Theta, Y, R)
print("Loss over random parameters: ", J)

# %%
maxiter = 200
epsilon = 1e-6
trained_params, Y_mean = fit(X, Theta, Y, R, maxiter, epsilon)

# %%
X = trained_params[: n_movies * n_features].reshape(n_movies, n_features)
Theta = trained_params[n_movies * n_features :].reshape(n_users, n_features)
predictions = predict(X, Theta, Y_mean)

# %%
movieList = open("data/dado3.txt", "r").read().split("\n")[:-1]
movieList

# %%
df = pd.DataFrame(np.hstack((predictions, np.array(movieList)[:, np.newaxis])))
df.sort_values(by=[0], ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# %%
print("Best recomendations:\n")
for i in range(10):
    print("Predicted ", round(float(df[0][i]), 1), " for index", df[1][i])
