import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, Lambda):
    X = params[: num_movies * num_features].reshape(num_movies, num_features)
    Theta = params[num_movies * num_features :].reshape(num_users, num_features)

    predictions = X @ Theta.T
    err = predictions - Y
    J = 1 / 2 * np.sum((err**2) * R)

    reg_X = Lambda / 2 * np.sum(Theta**2)
    reg_Theta = Lambda / 2 * np.sum(X**2)
    reg_J = J + reg_X + reg_Theta

    X_grad = err * R @ Theta
    Theta_grad = (err * R).T @ X
    grad = np.append(X_grad.flatten(), Theta_grad.flatten())

    reg_X_grad = X_grad + Lambda * X
    reg_Theta_grad = Theta_grad + Lambda * Theta
    reg_grad = np.append(reg_X_grad.flatten(), reg_Theta_grad.flatten())

    return J, grad, reg_J, reg_grad


def normalizeRatings(Y, R):
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
        cost, grad = cofiCostFunc(
            params, Y, R, num_users, num_movies, num_features, Lambda
        )[2:]

        X_grad = grad[: num_movies * num_features].reshape(num_movies, num_features)
        Theta_grad = grad[num_movies * num_features :].reshape(num_users, num_features)
        X = X - (alpha * X_grad)
        Theta = Theta - (alpha * Theta_grad)
        J_history.append(cost)

    paramsFinal = np.append(X.flatten(), Theta.flatten())
    return paramsFinal, J_history
