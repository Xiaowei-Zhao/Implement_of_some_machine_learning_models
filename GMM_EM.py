import numpy as np
import pandas as pd
import scipy.stats as st
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

X_train = pd.read_csv('Prob2_Xtrain.csv',header = None)
y_train = pd.read_csv('Prob2_ytrain.csv',header = None)


df=pd.concat([X_train, y_train],axis=1)
x1=df[df.iloc[:,-1]==1].iloc[:,0:-1]
x0=df[df.iloc[:,-1]==0].iloc[:,0:-1]


def init_GMM(X, k):
    # Initialize all covariance matrices to the empirical covariance of the data being modeled
    cov = [np.cov(X.T)]*k
    # Randomly initialize the means by sampling from a single multivariate Gaussian where the parameters are the mean
    # and covariance of the data being modeled
    mean = np.random.multivariate_normal(np.mean(X, axis = 0), cov[0],k)
    # Initialize the mixing weights to be uniform
    pi = [1/k]*k
    return pi,cov,mean


def E_step(x,pi,mean,cov,k):
    phi =  []
    denominator = 0
    for i in range(k):
        denominator += pi[i] * st.multivariate_normal.pdf(x,mean[i],cov[i],allow_singular=True)
    for i in range(k):
        numerator = st.multivariate_normal.pdf(x,mean[i],cov[i],allow_singular=True)
        phi.append(pi[i] * numerator / denominator)
    return phi


def M_step(x,phi,n,k):
    nk = np.sum(phi, axis = 1)
    pi_k = nk/n
    miu_k = np.dot(phi,x)/(nk.reshape(-1,1))
    cov = []
    for i in range(k):
        a=(x-miu_k[i]).T@np.diag(phi[i])
        temp=np.dot(a,(x-miu_k[i]))
        tempcov = temp/nk[i]
        cov.append(tempcov)
    return pi_k, miu_k, cov


def EM(x0,x1,k):    
    L1 = []    
    L0 = []
    pi1, cov1, mean1 = init_GMM(x1, k)
    pi0, cov0, mean0 = init_GMM(x0,k)
    for iterations in range(30):
            phi1 = E_step(x1,pi1,mean1,cov1,k)
            objective_function1 = 0
            for i in range(k):
                objective_function1 += np.sum(phi1[i])*math.log(pi1[i]) + phi1[i]@st.multivariate_normal.logpdf(x1,mean1[i],cov1[i],allow_singular=True)
            L1.append(objective_function1)
            pi1, mean1, cov1 = M_step(x1,phi1,x1.shape[0],k)
    
            phi0 = E_step(x0,pi0,mean0,cov0,k)
            objective_function0 = 0
            for i in range(k):
                objective_function0 += np.sum(phi0[i])*math.log(pi0[i]) + phi0[i]@st.multivariate_normal.logpdf(x0,mean0[i],cov0[i],allow_singular=True)
            L0.append(objective_function0)
            pi0, mean0, cov0 = M_step(x0,phi0,x0.shape[0],k) 
    return pi1,mean1,cov1,L1,pi0,mean0,cov0,L0

L_1 = []
L_0 = []
for i in range(10):
    k=3
    pi1,mean1,cov1,L1,pi0,mean0,cov0,L0 = EM(x0,x1,k)
    L_1.append(L1)
    L_0.append(L0)

plt.figure(figsize=(12,10))
plt.xlabel('Iterations')
plt.ylabel('Objective function L')
plt.title('GMM EM objective function L for class 1',fontsize=22)
for i in range(10):
    plt.plot(L_1[i])
plt.savefig('hw3_2_a_1.png',bbox_inches='tight')

plt.figure(figsize=(12,10))
plt.xlabel('Iterations')
plt.ylabel('Objective function L')
plt.xlim(5, 30)
plt.title('GMM EM objective function L for class 1(for iteration 5-30)',fontsize=22)
for i in range(10):
    plt.plot(L_1[i])
plt.savefig('hw3_2_a_2.png',bbox_inches='tight')

plt.figure(figsize=(12,10))
plt.xlabel('Iterations')
plt.ylabel('Objective function L')
plt.title('GMM EM objective function L for class 0',fontsize=22)
for i in range(10):
    plt.plot(L_0[i])
plt.savefig('hw3_2_a_3.png',bbox_inches='tight')


plt.figure(figsize=(12,10))
plt.xlabel('Iterations')
plt.ylabel('Objective function L')
plt.xlim(5, 30)
plt.title('GMM EM objective function L for class 0(for iteration 5-30)',fontsize=22)
for i in range(10):
    plt.plot(L_0[i])
plt.savefig('hw3_2_a_4.png',bbox_inches='tight')


from beautifultable import BeautifulTable
def predict_matrix(x1,x0,X_test,y_test,k):
    obj_1 = []
    obj_0 = []
    pi_1 = []
    mean_1 = []
    cov_1 = []
    pi_0 = []
    mean_0 = []
    cov_0 = []
    for run in range(10):
        pi1_temp,mean1_temp,cov1_temp,obj1_temp,pi0_temp,mean0_temp,cov0_temp,obj0_temp = EM(x0,x1,k)
        obj_1.append(obj1_temp[-1])
        obj_0.append(obj0_temp[-1])
        pi_1.append(pi1_temp)
        pi_0.append(pi0_temp)
        mean_1.append(mean1_temp)
        mean_0.append(mean0_temp)
        cov_1.append(cov1_temp)
        cov_0.append(cov0_temp)        
    idx1 = np.argmax(np.array(obj_1))
    idx0 = np.argmax(np.array(obj_0))
    pi1_best = pi_0[idx1]
    pi0_best = pi_0[idx0]
    mean1_best = mean_1[idx1]
    mean0_best = mean_0[idx0]
    cov1_best = cov_1[idx1]
    cov0_best = cov_0[idx0]
    
    cdf1 = 0
    cdf0 = 0
    for i in range(k):
        cdf1 += pi1_best[i]*st.multivariate_normal.pdf(X_test,mean1_best[i],cov1_best[i],allow_singular=True)
        cdf0 += pi0_best[i]*st.multivariate_normal.pdf(X_test,mean0_best[i],cov0_best[i],allow_singular=True)
    p0 = x0.shape[0]/X_train.shape[0]
    p1 = x1.shape[0]/X_train.shape[0]
    probability1 = cdf1 * p1
    probability0 = cdf0 * p0
    label = []
    for i in range(X_test.shape[0]):
        if probability1[i] >= probability0[i]:
            label.append(1)
        else:
            label.append(0)
    TP=0
    FP=0
    TN=0
    FN=0
    for i in range(len(label)):
        if y_test[i] == 1:
            if label[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label[i] == 1:
                FN += 1
            else:
                TN += 1
    table = BeautifulTable()
    table.column_headers = ["Ground Truth/Prediction", "1", "0"]
    table.append_row(["1", TP, FP])
    table.append_row(["0", FN, TN])
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    return table, accuracy


X_test = np.array(pd.read_csv('Prob2_Xtest.csv',header = None))
y_test = np.array(pd.read_csv('Prob2_ytest.csv',header = None))

table, accuracy = predict_matrix(x1,x0,X_test,y_test,1)
print(table)
print("The prediction accuracy for a 1-Gaussian GMM is {}".format(accuracy))

table, accuracy = predict_matrix(x1,x0,X_test,y_test,2)
print(table)
print("The prediction accuracy for a 2-Gaussian GMM is {}".format(accuracy))

table, accuracy = predict_matrix(x1,x0,X_test,y_test,3)
print(table)
print("The prediction accuracy for a 3-Gaussian GMM is {}".format(accuracy))

table, accuracy = predict_matrix(x1,x0,X_test,y_test,4)
print(table)
print("The prediction accuracy for a 4-Gaussian GMM is {}".format(accuracy))



