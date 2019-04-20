import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def update_C(data,miu):
    for i in miu.keys():
        data['distance_from_cluster{}'.format(i)] = (np.sqrt(np.power((data['x'] - miu[i][0][0]),2)+np.power((data['y'] - miu[i][0][1]),2)))
    # returns the index of the column where "distance_from_cluster#" has minimum value
    data['cluster'] = data.loc[:, ['distance_from_cluster{}'.format(i) for i in miu.keys()]].idxmin(axis=1)
    s='distance_from_cluster1'
    idx=s.index('1')
    data['cluster'] = data['cluster'].str.slice(idx)
    #data['color'] = data['cluster'].map(lambda x: colormap[x])
    return data

def update_C_with_color(data,miu,colormap):
    for i in miu.keys():
        data['distance_from_cluster{}'.format(i)] = (np.sqrt(np.power((data['x'] - miu[i][0][0]),2)+np.power((data['y'] - miu[i][0][1]),2)))
    # returns the index of the column where "distance_from_cluster#" has minimum value
    data['cluster'] = data.loc[:, ['distance_from_cluster{}'.format(i) for i in miu.keys()]].idxmin(axis=1)
    s='distance_from_cluster1'
    idx=s.index('1')
    data['cluster'] = data['cluster'].str.slice(idx)
    data['color'] = data['cluster'].map(lambda x: colormap[x])
    return data

def update_miu(data,miu):
    for i in miu.keys():
        miu[i][0][0] = np.mean(data[data['cluster'] == str(i)]['x'])
        miu[i][0][1] = np.mean(data[data['cluster'] == str(i)]['y'])
    return miu

def objective_function(data,miu):
    sum=0
    for i in range(data.shape[0]):
        for cluster in miu.keys():
            if data.cluster[i]==str(cluster):
                sum+=data['distance_from_cluster{}'.format(cluster)][i]**2
            else:
                sum+=0
    return sum

def K_means(data,k):
    colormap3 = {'1': 'red', '2': 'blue', '3': 'green'}
    colormap5 = {'1': 'red', '2': 'blue', '3': 'green', '4':'black', '5':'yellow'}
    miu = {
    i+1: 0.2*np.random.multivariate_normal(mean1, cov, 1) + 
          0.5*np.random.multivariate_normal(mean2, cov, 1) + 0.3*np.random.multivariate_normal(mean3, cov, 1)
    for i in range(k)}
    
    newdata=data
    obi=[]
    for i in range(20):
        newdata=update_C(newdata,miu)
        miu=update_miu(newdata,miu)
        obi.append(objective_function(newdata,miu))
    return obi,newdata, miu

def K_means2(data,k,colormap):
    miu = {
    i+1: 0.2*np.random.multivariate_normal(mean1, cov, 1) + 
          0.5*np.random.multivariate_normal(mean2, cov, 1) + 0.3*np.random.multivariate_normal(mean3, cov, 1)
    for i in range(k)}
    
    newdata=data
    obi=[]
    for i in range(20):
        newdata=update_C_with_color(newdata,miu,colormap)
        miu=update_miu(newdata,miu)
        obi.append(objective_function(newdata,miu))
    return obi,newdata, miu

# np.random.seed(123)
# mean1 = [0, 0]
# mean2 = [3, 0]
# mean3 = [0, 3]
# cov = [[1, 0], [0, 1]]
# data1 = np.random.multivariate_normal(mean1, cov, 500)
# data2 = np.random.multivariate_normal(mean2, cov, 500)
# data3 = np.random.multivariate_normal(mean3, cov, 500)
# data = 0.2*data1 + 0.5*data2 + 0.3*data3
mean1 = [0, 0]
mean2 = [3, 0]
mean3 = [0, 3]
cov = [[1, 0], [0, 1]]
data=[]
for i in range(500):  
    data1 = np.random.multivariate_normal(mean1, cov, 1).tolist()[0]
    data2 = np.random.multivariate_normal(mean2, cov, 1).tolist()[0]
    data3 = np.random.multivariate_normal(mean3, cov, 1).tolist()[0]
    choices = [data1,data2,data3]
    idx = np.random.choice(len(choices),p=[0.2,0.5,0.3])
    data.append(choices[idx])
data=np.array(data)
data=pd.DataFrame({'x': data[:,0],'y': data[:,1]})
L_4=[]
for k in [2,3,4,5]:
    obj, newdata, miu = K_means(data,k)
    L_4.append(obj)
plt.figure(figsize=(12,10))
plt.xlabel('Iterations')
plt.ylabel('K-Means objective function L')
plt.title('K-Means objective function L',fontsize=22)
for i in range(4):
    plt.plot(np.arange(20), L_4[i], label=i+2)
plt.legend(ncol=5, loc="lower right")
plt.savefig('hw3_1_1.png',bbox_inches='tight')


mean1 = [0, 0]
mean2 = [3, 0]
mean3 = [0, 3]
cov = [[1, 0], [0, 1]]
data=[]
for i in range(500):  
    data1 = np.random.multivariate_normal(mean1, cov, 1).tolist()[0]
    data2 = np.random.multivariate_normal(mean2, cov, 1).tolist()[0]
    data3 = np.random.multivariate_normal(mean3, cov, 1).tolist()[0]
    choices = [data1,data2,data3]
    idx = np.random.choice(len(choices),p=[0.2,0.5,0.3])
    data.append(choices[idx])
data=np.array(data)
data=pd.DataFrame({'x': data[:,0],'y': data[:,1]})
colormap3 = {'1': 'red', '2': 'blue', '3': 'green'}
colormap5 = {'1': 'red', '2': 'blue', '3': 'green', '4':'black', '5':'yellow'}
obj, newdata, miu = K_means2(data,3,colormap3)
plt.figure(figsize=(12,10))
plt.scatter(newdata['x'], newdata['y'], color=newdata['color'], alpha=0.3)
for i in miu.keys():
    plt.scatter(miu[i][0][0], miu[i][0][1], color=colormap3[str(i)],marker="*",s=200)
plt.savefig('hw3_1_2_1.png',bbox_inches='tight')


mean1 = [0, 0]
mean2 = [3, 0]
mean3 = [0, 3]
cov = [[1, 0], [0, 1]]
data=[]
for i in range(500):  
    data1 = np.random.multivariate_normal(mean1, cov, 1).tolist()[0]
    data2 = np.random.multivariate_normal(mean2, cov, 1).tolist()[0]
    data3 = np.random.multivariate_normal(mean3, cov, 1).tolist()[0]
    choices = [data1,data2,data3]
    idx = np.random.choice(len(choices),p=[0.2,0.5,0.3])
    data.append(choices[idx])
data=np.array(data)
data=pd.DataFrame({'x': data[:,0],'y': data[:,1]})
colormap3 = {'1': 'red', '2': 'blue', '3': 'green'}
colormap5 = {'1': 'red', '2': 'blue', '3': 'green', '4':'black', '5':'darkorange'}
obj, newdata, miu = K_means2(data,5,colormap5)
plt.figure(figsize=(12,10))
plt.scatter(newdata['x'], newdata['y'], color=newdata['color'], alpha=0.4)
for i in miu.keys():
    plt.scatter(miu[i][0][0], miu[i][0][1], color=colormap5[str(i)],marker="*",s=200)
plt.savefig('hw3_1_2_2.png',bbox_inches='tight')





