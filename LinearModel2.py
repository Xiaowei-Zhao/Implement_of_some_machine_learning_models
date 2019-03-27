import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x=pd.read_csv("X_train.csv",header=None)
y=pd.read_csv("y_train.csv",header=None)
x_test=pd.read_csv("X_test.csv",header=None)
y_test=pd.read_csv("y_test.csv",header=None)

# first order (p=1)
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
for lam in range(101):
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

data_test = pd.DataFrame({'cylinders':cylinders,'displacement':displacement,'horsepower':horsepower,
                     'weight':weight,'acceleration':acceleration,'year_made':year_made,'w_0':w_0})
RMSE=[]
for i in range(101):
    x_predict=x_test.dot(np.array(data_test.iloc[i]))
    rmse=np.sqrt(np.sum((y_test[0]-x_predict)**2)/len(y_test))
    RMSE.append(rmse)

# second order (p=2)
x2=x**2

x2=x2.iloc[:,0:-1]
x2=(x2-np.mean(x2))/np.std(x2)
one=pd.DataFrame(np.array([1]*len(x2)))
x2=pd.concat([x2, one], ignore_index=True,axis=1)

x1=x.iloc[:,0:6]
newx=pd.concat([x1, x2], ignore_index=True,axis=1)
u, s, vt = np.linalg.svd(newx,full_matrices=False)
v=np.transpose(vt)
ut=np.transpose(u)
cylinders=[] 
displacement=[]
horsepower=[]
weight=[]
acceleration=[]
year_made=[]
cylinders_2=[] 
displacement_2=[]
horsepower_2=[]
weight_2=[]
acceleration_2=[]
year_made_2=[]
w_0=[]
d_freedom=[]
for lam in range(101):
    s_lam=s/(s**2+lam)
    s_lam=np.diag(s_lam)
    Wrr=v.dot(s_lam).dot(ut).dot(y)
    cylinders.append(Wrr[0])
    displacement.append(Wrr[1])
    horsepower.append(Wrr[2])
    weight.append(Wrr[3])
    acceleration.append(Wrr[4])
    year_made.append(Wrr[5])
    cylinders_2.append(Wrr[6])
    displacement_2.append(Wrr[7])
    horsepower_2.append(Wrr[8])
    weight_2.append(Wrr[9])
    acceleration_2.append(Wrr[10])
    year_made_2.append(Wrr[11])
    w_0.append(Wrr[12])    
    d_freedom.append(np.sum((s**2)/(lam+s**2)))
    
x_test2=x_test**2

x_test2=x_test2.iloc[:,0:-1]
x_2=x**2
x_2=x_2.iloc[:,0:-1]
x_test2=(x_test2-np.mean(x_2))/np.std(x_2)
one=pd.DataFrame(np.array([1]*len(x_test2)))
x_test2=pd.concat([x_test2, one], ignore_index=True,axis=1)

x_test_2=x_test.iloc[:,0:6]
newx_test2=pd.concat([x_test_2, x_test2], ignore_index=True,axis=1)
data_test2 = pd.DataFrame({'cylinders':cylinders,'displacement':displacement,'horsepower':horsepower,
                     'weight':weight,'acceleration':acceleration,'year_made':year_made,'cylinders_2':cylinders_2,
                        'displacement_2':displacement_2,'horsepower_2':horsepower_2,
                     'weight_2':weight_2,'acceleration_2':acceleration_2,'year_made_2':year_made_2,'w_0':w_0})
RMSE2=[]
for i in range(101):
    x_predict=newx_test2.dot(np.array(data_test2.iloc[i]))
    rmse=np.sqrt(np.sum((y_test[0]-x_predict)**2)/len(y_test))
    RMSE2.append(rmse)
    
# Third order (p=2)
x3=x**3
x2=x2.iloc[:,0:-1]

x3=x3.iloc[:,0:-1]
x3=(x3-np.mean(x3))/np.std(x3)
one=pd.DataFrame(np.array([1]*len(x3)))
x3=pd.concat([x3, one], ignore_index=True,axis=1)

x1=x.iloc[:,0:6]
newx2=pd.concat([x1, x2, x3], ignore_index=True,axis=1)
u, s, vt = np.linalg.svd(newx2,full_matrices=False)
v=np.transpose(vt)
ut=np.transpose(u)
cylinders=[] 
displacement=[]
horsepower=[]
weight=[]
acceleration=[]
year_made=[]
cylinders_2=[] 
displacement_2=[]
horsepower_2=[]
weight_2=[]
acceleration_2=[]
year_made_2=[]
cylinders_3=[] 
displacement_3=[]
horsepower_3=[]
weight_3=[]
acceleration_3=[]
year_made_3=[]
w_0=[]
d_freedom=[]
for lam in range(101):
    s_lam=s/(s**2+lam)
    s_lam=np.diag(s_lam)
    Wrr=v.dot(s_lam).dot(ut).dot(y)
    cylinders.append(Wrr[0])
    displacement.append(Wrr[1])
    horsepower.append(Wrr[2])
    weight.append(Wrr[3])
    acceleration.append(Wrr[4])
    year_made.append(Wrr[5])
    cylinders_2.append(Wrr[6])
    displacement_2.append(Wrr[7])
    horsepower_2.append(Wrr[8])
    weight_2.append(Wrr[9])
    acceleration_2.append(Wrr[10])
    year_made_2.append(Wrr[11])
    cylinders_3.append(Wrr[12])
    displacement_3.append(Wrr[13])
    horsepower_3.append(Wrr[14])
    weight_3.append(Wrr[15])
    acceleration_3.append(Wrr[16])
    year_made_3.append(Wrr[17])
    w_0.append(Wrr[18])    
    d_freedom.append(np.sum((s**2)/(lam+s**2)))
    
x_test3=x_test**3
x_test2=x_test2.iloc[:,0:-1]

x_test3=x_test3.iloc[:,0:-1]
x_3=x**3
x_3=x_3.iloc[:,0:-1]
x_test3=(x_test3-np.mean(x_3))/np.std(x_3)
one=pd.DataFrame(np.array([1]*len(x_test3)))
x_test3=pd.concat([x_test3, one], ignore_index=True,axis=1)

x_test_2=x_test.iloc[:,0:6]
newx_test3=pd.concat([x_test_2, x_test2, x_test3], ignore_index=True,axis=1)
data_test3 = pd.DataFrame({'cylinders':cylinders,'displacement':displacement,'horsepower':horsepower,
                     'weight':weight,'acceleration':acceleration,'year_made':year_made,'cylinders_2':cylinders_2,
                        'displacement_2':displacement_2,'horsepower_2':horsepower_2,
                     'weight_2':weight_2,'acceleration_2':acceleration_2,'year_made_2':year_made_2,'cylinders_3':cylinders_3,
                        'displacement_3':displacement_3,'horsepower_3':horsepower_3,
                     'weight_3':weight_3,'acceleration_3':acceleration_3,'year_made_3':year_made_3,'w_0':w_0})
RMSE3=[]
for i in range(101):
    x_predict=newx_test3.dot(np.array(data_test3.iloc[i]))
    rmse=np.sqrt(np.sum((y_test[0]-x_predict)**2)/len(y_test))
    RMSE3.append(rmse)
    
lamb=list(range(101))
plt.figure(figsize=(16,10))
plt.xlabel('λ')
plt.ylabel('RMSE')
ax1 = plt.plot(lamb,RMSE,color='blue', label='First order(p=1)')
ax2 = plt.plot(lamb,RMSE2,color='red', label='Second order(p=2)')
ax3 = plt.plot(lamb,RMSE3,color='green', label='third order(p=3)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 16})
plt.grid()
plt.savefig('hw1_3_4.png',bbox_inches='tight')