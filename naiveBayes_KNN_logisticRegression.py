import numpy as np
import pandas as pd
from scipy.special import factorial
import matplotlib.pyplot as plt

data_x = pd.read_csv('X.csv',header=None)
data_y = pd.read_csv('y.csv',header=None)
data = pd.concat([data_x,data_y], axis=1)

# Problem 2(a)
def naiveBayes(test, train):
    train_one = train[train.iloc[:, -1] == 1]
    train_zero = train[train.iloc[:, -1] == 0]
    lambda_one = []
    lambda_zero = []
    for i in range(train.shape[1] - 1):
        lambda_temp_one = (sum(train_one.iloc[:, i]) + 1) / (train_one.shape[0] + 1)
        lambda_one.append(lambda_temp_one)
        lambda_temp_zero = (sum(train_zero.iloc[:, i]) + 1) / (train_zero.shape[0] + 1)
        lambda_zero.append(lambda_temp_zero)
    pi = train_one.shape[0] / (test.shape[0] + train.shape[0])
    p_y1 = pi
    p_y0 = 1 - pi
    prob_one = []
    prob_zero = []
    for i in range(test.shape[0]):
        a = p_y1 * np.exp(-1 * np.sum(lambda_one))
        b = np.prod(np.power(lambda_one, test.iloc[i, 0:-1]))
        c = np.prod(factorial(test.iloc[i, 0:-1]))
        prob_one.append(a * b / c)
        d = p_y0 * np.exp(-1 * np.sum(lambda_zero))
        e = np.prod(np.power(lambda_zero, test.iloc[i, 0:-1]))
        prob_zero.append(d * e / c)
    prob_one = np.array(prob_one)
    prob_zero = np.array(prob_zero)
    temp = (prob_one > prob_zero) * 1
    lambda_one = np.array(lambda_one)
    lambda_zero = np.array(lambda_zero)
    return temp, lambda_one, lambda_zero

from beautifultable import BeautifulTable
size=int(data.shape[0]/10)
cv=data.sample(frac=1)
TP=0
FP=0
TN=0
FN=0
lambda_1=np.zeros(cv.shape[1]-1)
lambda_0=np.zeros(cv.shape[1]-1)
for i in range(10):
    test=cv.iloc[i*size:(i+1)*size,:]
    train=cv.drop(index=test.index.tolist())
    temp,lambda_one,lambda_zero=naiveBayes(test,train)
    for j in range(len(temp)):
        if temp[j]==1:
            if test.iloc[j,-1]==1:
                TP+=1
            elif test.iloc[j,-1]==0:
                FP+=1
        if temp[j]==0:
            if test.iloc[j,-1]==1:
                FN+=1
            elif test.iloc[j,-1]==0:
                TN+=1
    lambda_1+=lambda_one
    lambda_0+=lambda_zero
lambda_1 = lambda_1/10
lambda_0 = lambda_0/10
table = BeautifulTable()
table.column_headers = ["Ground Truth/Prediction", "1", "0"]
table.append_row(["1", TP, FP])
table.append_row(["0", FN, TN])
print(table)
print("The prediction accuracy is {}%".format((TP+TN)*100/(TP+FP+TN+FN)))

# Problem 2(b)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 20))
axes[0].stem(np.arange(1,55),lambda_1, linefmt='-.')
axes[0].set_xlabel('Features',fontsize=15)
axes[0].set_ylabel('λ',fontsize=15)
axes[0].set_xticks(np.arange(1,55))
axes[0].set_title('Stem plot of the 54 Poisson parameters for class 1',fontsize=22)
axes[1].stem(np.arange(1,55),lambda_0, linefmt='-.')
plt.ylabel('λ',fontsize=15)
plt.xlabel('Features',fontsize=15)
plt.xticks(np.arange(1,55))
axes[1].set_title('Stem plot of the 54 Poisson parameters for class 0',fontsize=22)
plt.savefig('hw2_b.png',bbox_inches='tight')

# Problem 2(c)
def KNN(test,train,k):
    x_train=train.iloc[:,0:-1]
    finalresult=[]
    for i in range(test.shape[0]):
        mat=abs(x_train-test.iloc[i,0:-1])
        distance=mat.sum(axis=1)
        distance=np.array(distance)
        top_k_idx = np.argsort(distance)[:k]
        temp=train.iloc[top_k_idx,-1].values.tolist()
        if temp.count(1)>=temp.count(0):
            result=1
        else:
            result=0
        finalresult.append(result)
    a=np.array(test.iloc[:,-1].values.tolist())
    finalresult=np.array(finalresult)
    accuracy=sum(finalresult==a)/test.shape[0]
    return accuracy

size=int(data.shape[0]/10)
cv=data.sample(frac=1)
final_acc=[]
for k in np.arange(1,21):
    acc=[]
    for i in range(10):
        test=cv.iloc[i*size:(i+1)*size,:]
        train=cv.drop(index=test.index.tolist())
        accuracy=KNN(test,train,k)
        acc.append(accuracy)
    temp=sum(acc)/10
    final_acc.append(temp)
print(final_acc)

plt.figure(figsize=(16,10))
plt.xlabel('Parameter: K')
plt.ylabel('Prediction Accuracy')
plt.plot(np.arange(1,21),final_acc)
plt.xticks(np.arange(1,21))
plt.grid()
plt.title('Prediction accuracy as a function of k')
plt.savefig('hw2_c.png',bbox_inches='tight')

# Problem 2(d)
datanew_y=data_y[0].map({ 0 : -1,1:1})
data_x['My new column'] = 1
newdata =pd.concat([data_x,datanew_y], axis=1)


def Logistic(test, train, eta=0.01 / 4600):
    w = np.zeros(train.shape[1] - 1)
    L = []
    for i in range(1000):
        yxw = train.iloc[:, 0:-1] @ w * train.iloc[:, -1]
        e = np.exp(yxw)
        E = e / (1 + e)
        temp_L = np.sum(np.log(E))
        L.append(temp_L)
        a = np.array((1 - E) * train.iloc[:, -1])
        b = train.iloc[:, 0:-1] * a.reshape(-1, 1)
        summation = np.sum(b, axis=0)
        w = w + eta * summation
    return L

size=int(newdata.shape[0]/10)
cv=newdata.sample(frac=1)
L_10=[]
for i in range(10):
    test=cv.iloc[i*size:(i+1)*size,:]
    train=cv.drop(index=test.index.tolist())
    L=Logistic(test,train)
    L_10.append(L)

plt.figure(figsize=(12,10))
plt.xlabel('Steps')
plt.ylabel('Logistic regression objective training function L')
plt.title('Logistic regression objective training function',fontsize=22)
for i in range(10):
    plt.plot(np.arange(1000), L_10[i], label=i)
plt.legend(ncol=5, loc="lower right")
plt.savefig('hw2_d.png',bbox_inches='tight')

# Problem 2(e) and (f)
def newton(test,train):
    w=np.zeros(train.shape[1]-1)
    L=[]
    for i in range(1):
        yxw=train.iloc[:,0:-1]@w*train.iloc[:,-1]
        e=np.exp(yxw)
        E=e/(1+e)
        temp_L=np.sum(np.log(E))
        L.append(temp_L)
        a=np.array((1-E)*train.iloc[:,-1])
        b=train.iloc[:,0:-1]*a.reshape(-1,1)
        summation=np.sum(b,axis=0)
#        c=1+e
#         d=np.power(c,2)
#         f=np.array(-e/d)
        f=-1/(2+e+1/e)
        f=np.array(f)
        g=np.power(train.iloc[:,0:-1],2)
        h=g*f.reshape(-1,1)
        summation2=np.sum(h,axis=0)
        w = w-0.01*summation/summation2
    return L, w

def predict_class(x):
    if x>0.5:
        return 1
    else:
        return -1

TP=0
FP=0
TN=0
FN=0
size=int(newdata.shape[0]/10)
cv=newdata.sample(frac=1)
L_10=[]
for i in range(10):
    test=cv.iloc[i*size:(i+1)*size,:]
    train=cv.drop(index=test.index.tolist())
    L,w=newton(test,train)
    temp=test.iloc[:,0:-1]@w
    predict=np.exp(temp)/(1+np.exp(temp))
    predict_result=[]
    for i in predict:
        predict_result.append(predict_class(i))
    for j in range(len(predict_result)):
        if predict_result[j]==1:
            if test.iloc[j,-1]==1:
                TP+=1
            elif test.iloc[j,-1]==-1:
                FP+=1
        if predict_result[j]==-1:
            if test.iloc[j,-1]==1:
                FN+=1
            elif test.iloc[j,-1]==-1:
                TN+=1
    L_10.append(L)
table = BeautifulTable()
table.column_headers = ["Ground Truth/Prediction", "1", "-1"]
table.append_row(["1", TP, FP])
table.append_row(["-1", FN, TN])
print(table)
print("The prediction accuracy is {}%".format((TP+TN)*100/(TP+FP+TN+FN)))

plt.figure(figsize=(12,10))
plt.xlabel('Steps')
plt.ylabel('Logistic regression objective training function L')
plt.title('Logistic regression objective training function',fontsize=22)
for i in range(10):
    plt.plot(np.arange(100), L_10[i], label=i)
plt.legend(ncol=5, loc="lower right")
plt.savefig('hw2_e.png',bbox_inches='tight')