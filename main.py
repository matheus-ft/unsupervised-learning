from movie import (
    cofiCostFunc,
    normalizeRatings,
    gradientDescent,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


mat3 = loadmat("ex8_movies.mat")
mat4 = loadmat("ex8_movieParams.mat")
Y = mat3["Y"]
R = mat3["R"]
X = mat4["X"]
Theta = mat4["Theta"]

#%%
print("Average rating for movie 1 (Toy Story):",np.sum(Y[0,:]*R[0,:])/np.sum(R[0,:]),"/5")
