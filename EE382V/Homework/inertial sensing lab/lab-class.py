
# coding: utf-8

# In[4]:

import numpy as np


# In[5]:

dataset = np.loadtxt("gestures.csv", delimiter=",")


# In[6]:

dataset


# In[7]:

dataset.shape


# In[8]:

import matplotlib.pyplot as plt


# In[9]:

get_ipython().magic('matplotlib inline')


# In[10]:

plt.plot(dataset[:,0])


# In[11]:

plt.plot(dataset[:,1])


# In[13]:

plt.plot(dataset[:,2])


# In[14]:

plt.plot(dataset[:,3])


# In[15]:

dataset = dataset[:,1:]


# In[16]:

dataset


# In[17]:

dataset.shape


# In[18]:

dataset = dataset[1000:,:]


# In[19]:

dataset.shape


# In[21]:

plt.plot(dataset[:,2])


# In[22]:

from sklearn import preprocessing


# In[23]:

scaler = preprocessing.MinMaxScaler()


# In[24]:

dataset = scaler.fit_transform(dataset)


# In[25]:

plt.plot(dataset[:,2])


# In[ ]:




# In[27]:

frame_size = 1000
dataset_frame = dataset[0:frame_size, :]


# In[28]:

dataset_frame.shape


# In[29]:

plt.plot(dataset_frame[:,2])


# In[30]:

mean = np.mean(dataset_frame, axis=0)


# In[31]:

mean


# In[33]:

frame_size = 501
step_size = 250

for counter in range(0, len(dataset), step_size):
    
    dataset_frame = dataset[counter:counter+frame_size,:]
    
    mean = np.mean(dataset_frame, axis=0)
    var = np.var(dataset_frame, axis=0)
    
    data_frame_features = np.hstack((mean, var))
    
    if counter==0:
        dataset_feature_vector = data_frame_features
    else:
        dataset_feature_vector = np.vstack((dataset_feature_vector, data_frame_features))
    


# In[34]:

dataset_feature_vector.shape


# In[35]:

from sklearn import cluster


# In[45]:

k=2

kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(dataset_feature_vector)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

for i in range(k):
    ds = dataset_feature_vector[np.where(labels==i)]
    plt.plot(ds[:,0],ds[:,1],'o')
    
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)

