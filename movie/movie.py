import numpy as np


def compute_cost(X, Theta, Y, R):
    predictions = X @ Theta.T
    err = predictions - Y
    J = 1 / 2 * np.sum((err**2) * R)

    X_grad = err * R @ Theta
    Theta_grad = (err * R).T @ X
    grad = np.append(X_grad.flatten(), Theta_grad.flatten())

    return J, grad


def normalize_ratings(Y, R):
    m, n = Y.shape[0], Y.shape[1]
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros((m, n))

    for i in range(m):
        Ymean[i] = np.sum(Y[i, :]) / np.count_nonzero(R[i, :])
        Ynorm[i, R[i, :] == 1] = Y[i, R[i, :] == 1] - Ymean[i]

    return Ynorm, Ymean


def gradientDescent(
    initial_parameters,
    Y,
    R,
    num_users,
    num_movies,
    num_features,
    alpha,
    num_iters,
    Lambda,
):

    X = initial_parameters[: num_movies * num_features].reshape(
        num_movies, num_features
    )
    Theta = initial_parameters[num_movies * num_features :].reshape(
        num_users, num_features
    )

    J_history = []

    for _ in range(num_iters):
        params = np.append(X.flatten(), Theta.flatten())
        cost, grad = compute_cost(
            params, Y, R, num_users, num_movies, num_features, Lambda
        )[2:]

        X_grad = grad[: num_movies * num_features].reshape(num_movies, num_features)
        Theta_grad = grad[num_movies * num_features :].reshape(num_users, num_features)
        X = X - (alpha * X_grad)
        Theta = Theta - (alpha * Theta_grad)
        J_history.append(cost)

    paramsFinal = np.append(X.flatten(), Theta.flatten())
    return paramsFinal, J_history
