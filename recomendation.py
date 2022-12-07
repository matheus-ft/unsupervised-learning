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
movieList = open("data/dado3.txt", "r").read().split("\n")[:-1]

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
my_ratings = np.zeros((1682, 1))

my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[82] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print("Ratings of the new user")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print("Rating ", int(my_ratings[i]), " for the movie", movieList[i])

Y = np.hstack((my_ratings, Y))
R = np.hstack((my_ratings != 0, R))

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
maxiter = 1000
epsilon = 1e-6
trained_params, Y_mean = fit(X, Theta, Y, R, maxiter, epsilon)

# %%
X = trained_params[: n_movies * n_features].reshape(n_movies, n_features)
Theta = trained_params[n_movies * n_features :].reshape(n_users, n_features)
predictions = predict(X, Theta, Y_mean)

# %%
my_predictions = predictions[:, 0][:, np.newaxis]
df = pd.DataFrame(np.hstack((my_predictions, np.array(movieList)[:, np.newaxis])))
df.sort_values(by=[0], ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)

# %%
print("Best recomendations for the new user:\n")
for i in range(10):
    print("Predicted ", round(float(df[0][i]), 1), " for the movie", df[1][i])

# %%
