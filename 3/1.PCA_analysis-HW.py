
# coding: utf-8

# ## Homework 3
# 
# You will have to submit the following two completed ipython notebooks for this homework.
# 
# 1. PCA_analysis
# 2. Reconstruction

# In[2]:

get_ipython().magic(u'pylab inline')
#data_dir = "./Data/Weather/"
data_dir = "../../Data/Weather/"


# ### Downloading Pickled data from S3
# If `STAT.pickle` is not in the directory, get it using the following command

# In[3]:

get_ipython().system(u'curl -o $data_dir/STAT.pickle http://mas-dse-open.s3.amazonaws.com/Weather/STAT.pickle')


# ### Get the statistics from the Pickle File

# In[14]:

import pickle
STAT,STAT_description=pickle.load(open(data_dir+'/STAT.pickle','r'))


# In[31]:

STAT.keys()


# In[32]:

STAT_description


# In[17]:

Scalars=['mean','std','low1000','low100','high100','high1000']
for meas in STAT.keys():
    get_ipython().system(u"grep $meas '../../Data/Weather/ghcnd-readme.txt'")
    S=STAT[meas]
    for scalar in Scalars:
        print '%s:%f'%(scalar,S[scalar]),
    print


# ### Script for plotting yearly plots 

# In[18]:

def YearlyPlots(T,ttl='',yl='',xl='',y=None,x=None,size=(10,7), c=None):
    yearday=[i for i in range(1,366)]
    fig=figure(1,figsize=size,dpi=300)
    if shape(T)[0] != 365:
        raise ValueError("First dimension of T should be 365. Shape(T)="+str(shape(T)))
    months_name = ['Jan 1', 'Feb 1', 'Mar 1','Apr 1', 'May 1','Jun 1','Jul 1','Aug 1','Sep 1','Oct 1','Nov 1','Dec 1']
    xticks(np.array([1,32,60,91,121,152,182,213,244,274,305,335]), months_name[0:12], rotation=0)
    if c is not None:
        plot_date(yearday,T, '-',color=c);
    else:
        plot_date(yearday,T, '-', );
    # rotate and align the tick labels so they look better
    #fig.autofmt_xdate()
    ylabel(yl)
    xlabel(xl)
    if y is not None:
        ylim(y)
    if x is not None:
        xlim(x)
    grid()
    title(ttl)


# ### Plot the following 3 plots for each measurement:
# 
# 1. A histogram from the sample values (from SortedVals) restricted between low100 and high100 (By which we mean that any value larger or equal to low100 and smaller or equal to high100 is included).
# 2. Plot of mean and mean $\pm$ std
# 3. Number of measurements recorded each day

# In[19]:

def histogram(data, s):
    low = data['low100']
    high = data['high100']
    req = []
    temp = data['SortedVals']
    for i in temp:
        if i>= low and i<= high:
            req.append(i)
    hist(req, 100)
    title(s)
    
def meanstd(data,s):
    mean = data['Mean']
    YearlyPlots(mean,ttl=s,yl='',xl='',y=None,x=None,size=(10,7), c='r')
    std = np.power(data['Var'], 0.5)
    YearlyPlots(mean-std,ttl=s,yl='',xl='',y=None,x=None,size=(10,7), c='b')
    YearlyPlots(mean+std,ttl=s,yl='',xl='',y=None,x=None,size=(10,7), c='b')


# In[20]:

figure(figsize=(15,30))
offset=1
for meas in STAT.keys():
    subplot(6,3,offset)
    offset+=1
    ## Your code for Histogram
    histogram(STAT[meas], meas + ' restricted histogram')
    subplot(6,3,offset)
    offset+=1
    ## Your code for mean and mean +- std
    meanstd(STAT[meas], meas + ' mean +- std')
    subplot(6,3,offset)
    offset+=1
    ## Your code for number of measurements
    YearlyPlots(STAT[meas]['NE'],ttl=meas+' counts',yl='',xl='',y=None,x=None,size=(10,7), c='k')


# ### Plot the Number of measurements recorded each day for TMAX

# In[21]:

## Your code here
YearlyPlots(STAT['TMAX']['NE'],ttl='TMAX counts',yl='',xl='',y=None,x=None,size=(10,7), c='k')


# ### Extra Credit
# * Can you figure out what is the reason for these lower counts (especially at the beginning and end of the year and also the sudden dip at the end of each month)? Is it restricted to a subset of the stations? Suggest a way to remove this effect.
# 
# * Can you Explain the counts per day for "SNWD" ?
# 
# Provide your explanation in new markdown cells appended after this cell. Support your explanation
# using code cells and graphs. If you need more data that is available only in the full dataset in the cloud but not in the data you have downloaded, contact your TA.
# 

# In[22]:

def eigen(eigval, eigvec):
    sumi = np.sum(eigval)
    eigval /= sumi
    temp = np.cumsum(eigval[:9])
    temp = np.insert(temp,0,0)
    plot(temp)


# ### Plot the following 3 plots for each measurement:
# 
# 1. The percentage of variance explained by top-k eigen vectors for k between 1 to 9
# 2. Plot of mean and mean $\pm$ std
# 3. Plot of top 3 eigenvectors

# In[30]:

figure(figsize=(15,30))
offset=1
for meas in STAT.keys():
    subplot(6,3,offset)
    offset+=1
    ## Your code for percentage of variance explained
    eigvec = STAT[meas]['eigvec']
    eigen(STAT[meas]['eigval'], eigvec)
    subplot(6,3,offset)
    offset+=1
    ## Your code for mean and mean +- std
    meanstd(STAT[meas], meas + ' mean +- std')
    subplot(6,3,offset)
    offset+=1
    ## Your code for top-3 eigenvectors
    YearlyPlots(eigvec[:,0],ttl=meas+' Top 3 EigenVecs',yl='',xl='',y=None,x=None,size=(10,7), c='r')
    YearlyPlots(eigvec[:,1],ttl=meas+' Top 3 EigenVecs',yl='',xl='',y=None,x=None,size=(10,7), c='g')
    YearlyPlots(eigvec[:,2],ttl=meas+' Top 3 EigenVecs',yl='',xl='',y=None,x=None,size=(10,7), c='b')

