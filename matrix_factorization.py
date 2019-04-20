import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt


train = pd.read_csv('Prob3_ratings.csv',header = None)
train.columns = ['users', 'movies', 'score']


N1 = max(train['users'])
N2 = max(train['movies'])

M = coo_matrix((np.array(train['score']), (np.array(train['users'])-1, np.array(train['movies'])-1)), shape=(N1, N2)).toarray()


def get_idx(M,N1,N2):
    ui = []
    vi = []
    for i in range(N1):
        movie = []
        for j in range(N2):
            if M[i,j] != 0:
                movie.append(j)
        ui.append(movie)
        
    for i in range(N2):
        user = []
        for j in range(N1):
            if M[j,i] != 0:
                user.append(j)
        vi.append(user)
    return ui,vi


def Matrix_factorization(M,N1,N2):
    I = np.eye(10)
    L = []
    U = []
    V = []
    for simulation in range(10):
        L_temp = []
        # For each run, initialize your ui and vj vectors as N(0; I) random vectors
        u = np.random.multivariate_normal(np.zeros(10), I, N1)
        v = np.random.multivariate_normal(np.zeros(10), I, N2)

        user_idx,movie_idx = get_idx(M,N1,N2)
        # Train the model on the larger training set for 100 iterations
        for iterations in range(100):
            for i in range(N1):
                vj = v[user_idx[i]]
                inverse = np.linalg.inv(0.25*I + vj.T@vj)
                temp = M[i,user_idx[i]].reshape(-1,1)
                mv = vj.T@temp
                u[i] = np.dot(inverse,mv).reshape(1,-1)

            for i in range(N2):
                ui = u[movie_idx[i]]
                inverse = np.linalg.inv(0.25*I + ui.T@ui)
                temp = M[movie_idx[i],i].reshape(-1,1)
                mu = ui.T@temp
                v[i] = np.dot(inverse,mu).reshape(1,-1)
            # For Mij is measured
            M_minus_uv_measured = (M - u@v.T)[M != 0]
            obj = -1*sum(M_minus_uv_measured**2)/0.5 - 0.5*np.sum(u**2) - 0.5*np.sum(v**2)
            L_temp.append(obj)
        L.append(L_temp)
        U.append(u)
        V.append(v)
    return L,U,V


L,U,V = Matrix_factorization(M,N1,N2)


plt.figure(figsize=(12,10))
plt.xlabel('Iterations')
plt.ylabel('Objective function L')
plt.title('The log joint likelihood for iterations 2 to 100 for each run',fontsize=22)
plt.xlim(2, 100)
plt.ylim(-120000, -90000)
for i in range(10):
    plt.plot(L[i])
plt.savefig('hw3_3_a.png',bbox_inches='tight')



test =  pd.read_csv('Prob3_ratings_test.csv',header = None)
test.columns = ['users', 'movies', 'score']
training_objective_function = []
for i in range(10):
    training_objective_function.append(L[i][-1])  
RMSE = [] 
for simulation in range(10):
    MSE = 0
    for i in range(test.shape[0]):
        u=U[simulation][test['users'][i] - 1]
        v=V[simulation][test['movies'][i] - 1]
        MSE += (test['score'][i] - u@v)**2
    RMSE.append(np.sqrt(MSE/test.shape[0]))   
table = pd.DataFrame({'Object Function':training_objective_function, 'RMSE':RMSE})
table.sort_values('Object Function',ascending=False)


with open('Prob3_movies.txt', encoding='UTF-8') as file:
    movie = file.read()
movies = movie.split('\n')


idx = np.argmax(np.array(training_objective_function))
v = V[idx]
temp=pd.DataFrame({'Movies':movies})
Star_Wars=temp[temp.Movies.str.contains('Star Wars')].iloc[0,0]
Star_Wars_index=temp.index[temp.Movies == Star_Wars].tolist()[0]
My_Fair_Lady=temp[temp.Movies.str.contains('My Fair Lady')].iloc[0,0]
My_Fair_Lady_index=temp.index[temp.Movies == My_Fair_Lady].tolist()[0]
GoodFellas=temp[temp.Movies.str.contains('GoodFellas')].iloc[0,0]
GoodFellas_index=temp.index[temp.Movies == GoodFellas].tolist()[0]

distance_Star_Wars = np.sqrt(np.sum((v-v[Star_Wars_index])**2, axis = 1))
index=distance_Star_Wars.argsort()[:11]
distance=[]
name=[]
for i in index:
    name.append(movies[i])
    distance.append(distance_Star_Wars[i])
starwar = pd.DataFrame({'Closest movies':name, 'Distance':distance})
print(starwar)


distance_My_Fair_Lady = np.sqrt(np.sum((v-v[My_Fair_Lady_index])**2, axis = 1))
index=distance_My_Fair_Lady.argsort()[:11]
distance=[]
name=[]
for i in index:
    name.append(movies[i])
    distance.append(distance_My_Fair_Lady[i])
fairlady = pd.DataFrame({'Closest movies':name, 'Distance':distance})
print(fairlady)



distance_GoodFellas = np.sqrt(np.sum((v-v[GoodFellas_index])**2, axis = 1))
index=distance_GoodFellas.argsort()[:11]
distance=[]
name=[]
for i in index:
    name.append(movies[i])
    distance.append(distance_GoodFellas[i])
good = pd.DataFrame({'Closest movies':name, 'Distance':distance})
print(good)




