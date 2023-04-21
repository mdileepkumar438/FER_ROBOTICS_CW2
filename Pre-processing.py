import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#Load dataset from CSV file
data = pd.read_csv('/Users/dileep/Desktop/CSV_Dataset/fer2013.csv')

#Define image width and height
width, height = 48, 48


#Extract image pixels from dataset and convert them into numpy array
datapoints = data['pixels'].tolist()

#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

#getting labels for training
y = pd.get_dummies(data['emotion']).values

#storing them using numpy
np.save('./model/fdataX', X)
np.save('./model/flabels', y)


#Print information about the preprocessed data
print("Preprocessing Done")
print("Number of Features: "+str(len(X[0])))
print("Number of Labels: "+ str(len(y[0])))
print("Number of examples in dataset:"+str(len(X)))
print("X,y stored in fdataX.npy and flabels.npy respectively")