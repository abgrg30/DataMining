
# coding: utf-8

# In[4]:

get_ipython().magic(u'pwd')
get_ipython().magic(u'pylab inline')


# In[5]:

data_dir = "../../Data/Weather/"
get_ipython().magic(u'matplotlib inline')
get_ipython().system(u'mkdir -p $data_dir')
get_ipython().system(u'ls $data_dir')


# ### Downloading Pickled data from S3
# If `SampleStations.pickle` is not in the directory, get it using the following command

# In[6]:

get_ipython().system(u'curl -o $data_dir/SampleStations.pickle http://mas-dse-open.s3.amazonaws.com/Weather/SampleStations.pickle')


# ### Plot Reconstructions
# 
# From each measurement, we take 2 sample stations with low residual error and 2 sample stations with high residual error and plot the following:
# 
# * Original Data Vector
# * Mean Vector
# * Reconstructed Data Vector using mean and top 1 eigen vectors
# * Reconstructed Data Vector using mean and top 2 eigen vectors
# * Reconstructed Data Vector using mean and top 3 eigen vectors

# ### Read and Preprocess data
# 
# Read data from the pickle files `SampleStations.pickle` and `STAT.pickle`
# 
# * `SampleStations.pickle` contains information about the 6 measurements taken from some sample stations. You have been provided the code to process this file. It converts the seemingly complicated structure into a list of lists where each sublist has the following information:
# 
# `['station','measurement','year','1', '2', '3', ...... , '365']`
# 
# *  station - The station ID
# *  measurement - One of the 6 types of measurements
# *  year - The year in which the measurements were recorded
# *  1-365 - The actual value of measurement for each day of the year
# 
# 
# 
# * `STAT.pickle` contains statistics about the weather data for each of the 6 measurements and its description.

# In[7]:

import pickle
Data=pickle.load(open('../../Data/Weather/SampleStations.pickle','r'))
STAT,STAT_description=pickle.load(open('../../Data/Weather/STAT.pickle','r'))


# In[8]:

FlatData=[]
for station in Data:
    stationname=station[0]
    for measurements in station[1]:
        measurement,year=measurements[0]
        yeardata=list(measurements[1])
        rowData=[stationname]+[measurement]+[year]+yeardata
        FlatData.append(rowData)


# In[9]:

import pandas as pd

frameheader=['station','measurement','year']+range(1,366)
df=pd.DataFrame(FlatData,columns=frameheader)

m_df={}
for m in ['TMIN', 'TOBS', 'TMAX', 'SNOW', 'SNWD', 'PRCP']:
    t_df=df[df['measurement']==m]
    m_df[m]=t_df    


# ### Define Reconstruction Function
# 
# You need to plot reconstructions for two stations with low reconstruction error and two stations with high reconstruction error. To do this, you will need to do the following:
# 
# 1. Calculate the reconstruction error for the data vector reconstructed using mean and top-3 eigenvectors.
# 2. Remove the ones for which reconstruction error is NaN.
# 3. Choose two with the lowest and two with the highest reconstruction error for plotting.

# In[180]:

def create_reconstructions(m):  
    
    ## Put your code for computating reconstructions here    

    eigvec3 = STAT[m]['eigvec'][:,0:3]
    mean = STAT[m]['Mean']
    original = m_df[m].as_matrix(range(1,366))
    repmean = np.tile(mean, (original.shape[0],1)).T
    
    Reconstructed = np.zeros((4, original.shape[1], original.shape[0]))
    
    projection = np.dot(eigvec3[:,0].T, original.T)
    Reconstructed[1] = np.outer(eigvec3[:,0], projection)
    
    projection = np.dot(eigvec3[:,0:2].T, original.T)
    Reconstructed[2] = np.dot(eigvec3[:,0:2], projection)
    
    projection = np.dot(eigvec3.T, original.T)
    Reconstructed[3] = np.dot(eigvec3, projection)
    
    error = np.nansum((Reconstructed[3] - repmean)**2, axis=0)

    error[error==NaN] = float('inf')
    argmin1 = np.argmin(error)
    val = error[argmin1]
    error[argmin1] = float('inf')
    argmin2 = np.argmin(error)
    error[argmin1] = val    

    error[error==float('inf')] = float('-inf')
    argmax1 = np.argmax(error)
    val = error[argmax1]
    error[argmax1] = float('-inf')
    argmax2 = np.argmax(error)
    error[argmax1] = val

    error[error==float('-inf')] = NaN

    lower = []
    upper = []

    lower.append(argmin1)
    lower.append(argmin2)
    upper.append(argmax1)
    upper.append(argmax2)       
        
    
    yeardays=[i for i in (1,366)]
    plt.figure(figsize=(20,30),dpi=300)
    j=1
    c=0
    for l in lower:
        subplot(4,2,j)
        j+=1
        c+=1
        plot(original[l])
        plot(STAT[m]['Mean'])
        plot(Reconstructed[1][:,l])
        plot(Reconstructed[2][:,l])
        plot(Reconstructed[3][:,l])
        title('#' + str(c) + ' Sample for ' + m + ' (low residual error)')
        xlim([0,365])
        legend(['original','Mean','1','2','3'],loc=2)
    
    c=0
    for l in upper:
        subplot(4,2,j)
        j+=1
        c+=1
        plot(original[l])
        plot(STAT[m]['Mean'])
        plot(Reconstructed[1][:,l])
        plot(Reconstructed[2][:,l])
        plot(Reconstructed[3][:,l])
        title('#' + str(c) + ' Sample for ' + m + ' (high residual error)')
        xlim([0,365])
        legend(['original','Mean','1','2','3'],loc=2)


# In[181]:

for m in ['TMAX','SNWD']:
    print 'Reconstruction Plots for '+ m
    create_reconstructions(m)


# In[12]:

for m in ['TMAX','SNWD']:
    print 'Reconstruction Plots for '+ m
    create_reconstructions(m)

