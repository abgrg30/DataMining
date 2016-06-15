
# coding: utf-8

# In[ ]:

# Name: ABHINAV GARG
# Email: abgarg@ucsd.edu
# PID: A53095668
from pyspark import SparkContext
sc = SparkContext()
# Your program here


# In[1]:

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from string import split,strip
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils


# In[2]:

# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/HIGGS/HIGGS.csv'
inputRDD=sc.textFile(path)


# In[3]:

input_sampled = inputRDD.sample(False,0.1, seed=255)
Data=input_sampled.map(lambda line: [float(x) for x in line.split(',')]).map(lambda V:LabeledPoint(V[0], V[1:]))


# In[4]:

Data.cache()
(trainingData,testData)=Data.randomSplit([0.7,0.3], seed=255)


# In[ ]:

depth=10
model=GradientBoostedTrees.trainClassifier(trainingData,categoricalFeaturesInfo={},                                           numIterations=25,learningRate=0.2,maxDepth=depth)
errors={}
errors[depth]={}
dataSets={'train':trainingData,'test':testData}
for name in dataSets.keys():  # Calculate errors on train and test sets
    data=dataSets[name]
    Predicted=model.predict(data.map(lambda x: x.features))
    LabelsAndPredictions=data.map(lambda x : x.label).zip(Predicted)
    Err = LabelsAndPredictions.filter(lambda (v,p):v != p).count()/float(data.count())
    errors[depth][name]=Err
print depth,errors[depth]

