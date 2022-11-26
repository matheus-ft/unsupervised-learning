import numpy as np
import numpy.random as npr
from scipy.optimize import minimize, OptimizeResult


def compute_cost(
    X: np.ndarray, Theta: np.ndarray, Y: np.ndarray, R: np.ndarray
) -> float:
    params = np.append(X.flatten(), Theta.flatten())
    n_movies, n_features = X.shape
    n_users = Theta.shape[0]
    args = (Y, R, n_movies, n_users, n_features)
    return _loss(params, *args)


def normalize_ratings(Y: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, n = Y.shape[0], Y.shape[1]
    Y_mean = np.zeros((m, 1))
    Y_norm = np.zeros((m, n))
    for i in range(m):
        Y_mean[i] = np.sum(Y[i, :]) / np.count_nonzero(R[i, :])
        Y_norm[i, R[i, :] == 1] = Y[i, R[i, :] == 1] - Y_mean[i]
    return Y_norm, Y_mean


def init_random_matrix(m: int, n: int) -> np.ndarray:
    epi = (6 ** (1 / 2)) / (m + n - 1) ** (1 / 2)
    M = npr.rand(m, n) * (2 * epi) - epi
    return M


def _infer_ratings(X: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    return X @ Theta.T


def predict(X: np.ndarray, Theta: np.ndarray, Y_mean: np.ndarray) -> np.ndarray:
    return _infer_ratings(X, Theta) + Y_mean


def _error(X: np.ndarray, Theta: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return _infer_ratings(X, Theta) - Y


def _loss(params: np.ndarray, *args) -> float:
    """Computes the loss function of the model.

    Parameters
    ----------
    params : np.ndarray
        1-D array of all features and their weights concatenated

    *args : sequence of arguments
        Must be (Y, R, n_m, n_u, n_f)

    Returns
    -------
    float
        value of the loss function
    """
    Y, R, n_movies, n_users, n_features = args
    X = params[: n_movies * n_features].reshape(n_movies, n_features)
    Theta = params[n_movies * n_features :].reshape(n_users, n_features)
    error_vec = _error(X, Theta, Y)
    J = np.sum(error_vec**2 * R) / 2
    return J


def _gradient(params: np.ndarray, *args) -> np.ndarray:
    """Gradient of the hypothesis function.

    Parameters
    ----------
    params : np.ndarray
        1-D array of all features and their weights concatenated

    *args : sequence of arguments
        Must be (Y, R, n_m, n_u, n_f)

    Returns
    -------
    np.ndarray
        gradient vector (as 1-d array)
    """
    Y, R, n_movies, n_users, n_features = args
    X = params[: n_movies * n_features].reshape(n_movies, n_features)
    Theta = params[n_movies * n_features :].reshape(n_users, n_features)
    error_vec = _error(X, Theta, Y)
    D_X = error_vec * R @ Theta
    D_Theta = (error_vec * R).T @ X
    D = np.append(D_X.flatten(), D_Theta.flatten())
    return D


def fit(
    X: np.ndarray,
    Theta: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    maxiter: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    Y, Y_mean = normalize_ratings(Y, R)
    optimal = _conjugate_gradient(X, Theta, Y, R, maxiter, epsilon)
    if optimal.success:
        return optimal.x, Y_mean
    raise Exception(
        f"Conjugate gradient method (via `scipy.optimize.minimize`) failed with the following message:\n{optimal.message}"
    )


def _conjugate_gradient(
    X: np.ndarray,
    Theta: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    maxiter: int,
    epsilon: float,
) -> OptimizeResult:
    params = np.append(X.flatten(), Theta.flatten())
    n_movies, n_features = X.shape
    n_users = Theta.shape[0]
    args = (Y, R, n_movies, n_users, n_features)
    opts = {"maxiter": maxiter, "disp": True}
    return minimize(
        _loss, params, args, method="CG", jac=_gradient, tol=epsilon, options=opts
    )
