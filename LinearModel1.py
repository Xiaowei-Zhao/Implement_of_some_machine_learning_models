import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=pd.read_csv("X_train.csv",header=None)
y=pd.read_csv("y_train.csv",header=None)
u, s, vt = np.linalg.svd(x,full_matrices=False)
v=np.transpose(vt)
ut=np.transpose(u)
cylinders=[] 
displacement=[]
horsepower=[]
weight=[]
acceleration=[]
year_made=[]
w_0=[]
d_freedom=[]
for lam in range(5001):
    s_lam=s/(s**2+lam)
    s_lam=np.diag(s_lam)
    Wrr=v.dot(s_lam).dot(ut).dot(y)
    cylinders.append(Wrr[0])
    displacement.append(Wrr[1])
    horsepower.append(Wrr[2])
    weight.append(Wrr[3])
    acceleration.append(Wrr[4])
    year_made.append(Wrr[5])
    w_0.append(Wrr[6])
    d_freedom.append(np.sum((s**2)/(lam+s**2)))
    
plt.figure(figsize=(16,10))
plt.xlabel('df(λ)')
plt.ylabel('w')
data = pd.DataFrame({'d_freedom':d_freedom,'cylinders':cylinders,'displacement':displacement,'horsepower':horsepower,
                     'weight':weight,'acceleration':acceleration,'year_made':year_made,'w_0':w_0})
ax1 = plt.plot(data.d_freedom,data.cylinders,color='blue', label='cylinders')
ax2 = plt.plot(data.d_freedom,data.displacement,color='red', label='displacement')
ax3 = plt.plot(data.d_freedom,data.horsepower,color='black', label='horsepower')
ax4 = plt.plot(data.d_freedom,data.weight,color='green', label='weight')
ax5 = plt.plot(data.d_freedom,data.acceleration,color='goldenrod', label='acceleration')
ax6 = plt.plot(data.d_freedom,data.year_made,color='purple', label='year_made')
ax7 = plt.plot(data.d_freedom,data.w_0,color='brown', label='w_0')

plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 16})
plt.savefig('hw1_3_1.png',bbox_inches='tight')

x_test=pd.read_csv("X_test.csv",header=None)
y_test=pd.read_csv("y_test.csv",header=None)
data_test = pd.DataFrame({'cylinders':cylinders,'displacement':displacement,'horsepower':horsepower,
                     'weight':weight,'acceleration':acceleration,'year_made':year_made,'w_0':w_0})
RMSE=[]
for i in range(51):
    x_predict=x_test.dot(np.array(data_test.iloc[i]))
    rmse=np.sqrt(np.sum((y_test[0]-x_predict)**2)/len(y_test))
    RMSE.append(rmse)
lamb=list(range(51))
plt.figure(figsize=(16,10))
plt.xlabel('λ')
plt.ylabel('RMSE')
ax = plt.plot(lamb,RMSE)
plt.grid()
plt.savefig('hw1_3_3.png',bbox_inches='tight')