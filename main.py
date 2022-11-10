from movie import (
    cofiCostFunc,
    normalizeRatings,
    gradientDescent,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


mat3 = loadmat("data/ex8_movies.mat")
mat4 = loadmat("data/ex8_movieParams.mat")
Y = mat3["Y"]
R = mat3["R"]
X = mat4["X"]
Theta = mat4["Theta"]

#%%
print("Average rating for movie 1 (Toy Story):",np.sum(Y[0,:]*R[0,:])/np.sum(R[0,:]),"/5")
#%%
plt.figure(figsize=(8,16))
plt.imshow(Y)
plt.xlabel("Users")
plt.ylabel("Movies")
#%%
num_users, num_movies, num_features = 4,5,3
X_test = X[:num_movies,:num_features]
Theta_test= Theta[:num_users,:num_features]
Y_test = Y[:num_movies,:num_users]
R_test = R[:num_movies,:num_users]
params = np.append(X_test.flatten(),Theta_test.flatten())

J, grad = cofiCostFunc(params, Y_test, R_test, num_users, num_movies, num_features, 0)[:2]
print("Custo sobre os parâmetros carregados:",J)

J2, grad2 = cofiCostFunc(params, Y_test, R_test, num_users, num_movies, num_features, 1.5)[2:]
print("Custo sobre os parâmetros carregados (lambda = 1.5):",J2)
#%%
movieList = open("data/dado3.txt","r").read().split("\n")[:-1]
movieList
#%%
my_ratings = np.zeros((1682,1))

my_ratings[0] = 4 
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[82]= 4
my_ratings[225] = 5
my_ratings[354]= 5

print("Notas do novo usuário:\n")
for i in range(len(my_ratings)):
    if my_ratings[i]>0:
        print("Nota",int(my_ratings[i]),"para o índice",movieList[i])
        
#%%
Y = np.hstack((my_ratings,Y))
R = np.hstack((my_ratings!=0,R))

Ynorm, Ymean = normalizeRatings(Y, R)
#%%
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.append(X.flatten(),Theta.flatten())
Lambda = 10

paramsFinal, J_history = gradientDescent(initial_parameters,Ynorm,R,num_users,num_movies,num_features,0.001,20,Lambda)
#%%
plt.plot(J_history)
plt.xlabel("Iterações")
plt.ylabel("$J(\Theta)$")
plt.title("Função de Custo usando Gradiente Descendente")
#%%
X = paramsFinal[:num_movies*num_features].reshape(num_movies,num_features)
Theta = paramsFinal[num_movies*num_features:].reshape(num_users,num_features)

p = X @ Theta.T
my_predictions = p[:,0][:,np.newaxis] + Ymean
#%%
import pandas as pd
df = pd.DataFrame(np.hstack((my_predictions,np.array(movieList)[:,np.newaxis])))
df.sort_values(by=[0],ascending=False,inplace=True)
df.reset_index(drop=True,inplace=True)
#%%
print("Melhores recomendações para você:\n")
for i in range(10):
    print("Nota predita",round(float(df[0][i]),1)," para o índice",df[1][i])