
# coding: utf-8

# In[ ]:

# Name: ABHINAV GARG
# Email: abgarg@ucsd.edu
# PID: A53095668
from pyspark import SparkContext
sc = SparkContext()
# Your program here


# In[6]:

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from string import split,strip
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils


# In[7]:

# Read the file into an RDD
# If doing this on a real cluster, you need the file to be available on all nodes, ideally in HDFS.
path='/covtype/covtype.data'
inputRDD=sc.textFile(path)


# In[8]:

Label=2.0
Data=inputRDD.map(lambda line: [float(x) for x in line.split(',')])    .map(lambda V:LabeledPoint(1 if V[-1]==Label else 0, V[:-1]))


# In[9]:

Data.cache()
(trainingData,testData)=Data.randomSplit([0.7,0.3], seed=255)


# In[ ]:

depth=10
d = dict()
for i in xrange(10,54):
    d[i] = 2
model=GradientBoostedTrees.trainClassifier(trainingData,categoricalFeaturesInfo=d,                                           numIterations=10,learningRate=0.5,maxDepth=depth)
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

