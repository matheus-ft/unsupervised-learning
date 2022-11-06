import numpy as np
import numpy.linalg as npla
import numpy.random as npr


def pca(data: np.ndarray, variance=0.99) -> tuple[np.ndarray, np.ndarray, int, int]:
    m = data.shape[0]
    Sigma = data.T @ data / m
    U, S, _ = npla.svd(Sigma, hermitian=True)  # real and symmetric == hermitian
    r = len(S)
    full_var = sum(S)
    for k in range(r):
        reduced_var = sum(S[:k])
        if reduced_var / full_var >= variance:
            return U, S, k, r
    return U, S, r, r


def zscore(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = np.mean(data, axis=0)  # average over the rows (i.e. examples)
    sigma = np.std(data, axis=0)
    normalized = (data - mu) / sigma
    return normalized, mu, sigma


def compress(original: np.ndarray, U_r: np.ndarray) -> np.ndarray:
    compressed = original @ U_r
    return compressed


def decompress(compressed: np.ndarray, U_r: np.ndarray) -> np.ndarray:
    approximated = compressed @ U_r.T
    return approximated


def img_from_vector(img: np.ndarray) -> np.ndarray:
    pixels = len(img)
    img_size = int(np.sqrt(pixels))
    shape = (img_size, img_size)
    image = img.reshape(shape, order="F")  # will throw an error if sqrt is not integer
    return image


def random_img(imgs_in_rows: np.ndarray) -> np.ndarray:
    rows, pixels = imgs_in_rows.shape
    rand_ind = npr.randint(0, rows)
    img_size = int(np.sqrt(pixels))
    image = imgs_in_rows[rand_ind].reshape((img_size, img_size), order="F")
    return image
