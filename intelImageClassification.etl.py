
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.image as img
from PIL import Image
import collections
from keras.utils import to_categorical
from random import randint
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[3]:


# paths for training and test data
trainPath = "ibm-ads-capstone/input/seg_train/"
testPath = "ibm-ads-capstone/input/seg_test/"


# In[4]:


# define a function to load images
def load_image(infilename):
    image = Image.open(infilename)
    image.load()
    return image


# In[5]:


# define etl function
def etl(folder):
    
    subFolders = os.listdir(folder)
    
    images = []
    X_data = []
    y_labels = []
    
    # read in images, convert to arrays, resize if necessary and append labels
    for subFolder in subFolders:
        
        subFolderPath = folder + subFolder + "/"
        print("Reading in images from " + subFolderPath)
        
        for file in os.listdir(subFolderPath):
            
            # read in image
            image = load_image(subFolderPath + file)
            
            # append image
            images.append(image)
            
            # convert to array
            array = np.asarray(image, dtype="int32")
            
            # resize if necessary
            if array.shape != (150, 150, 3):
                resized = image.resize((150, 150), Image.LANCZOS)
                array = np.asarray(resized, dtype="int32")

            # append data
            X_data.append(array)
            
            # append labels
            y_labels.append(subFolder)
    
    # Checks
    print("Number of data items = " + str(len(X_data)))
    print("Number of labels = " + str(len(y_labels)))
    print("Label frequencies: " + str(collections.Counter(y_labels)))
    
    # One-hot encode of labels
    
    # get number of classes
    classes = len(set(y_labels))

    # create dictionary which maps class to an integer label
    classDict = dict((label, counter) for counter, label in enumerate(list(set(y_labels))))
    print(classDict)

    # convert labels to integer values
    y_values = []
    for label in y_labels:
        y_values.append(classDict[label])

    # one-hot encode label values    
    y_onehot = to_categorical(y_values, classes)
    
    # Convert lists to arrays
    X_array = np.array(X_data)
    y_array = np.array(y_onehot)
    
    # Check array shapes
    print("X array shape = " + str(X_array.shape))
    print("y array shape = " + str(y_array.shape))
    print("y column totals = " + str(np.sum(y_array, axis=0)))
    
    # print out images and labels for 20 random records to make sure they are still aligned
    f, ax = plt.subplots(5, 4, figsize=(20, 20))
    for i in range(5):
        for j in range(4):
            r = randint(0,len(images))
            ax[i, j].imshow(images[r])
            ax[i, j].set_title(y_labels[r])
            ax[i, j].axis('off')
    
    # return as arrays
    return X_array, y_array, images, y_labels


# In[6]:


# Read in training data
X_train, y_train, images_train, labels_train = etl(trainPath)


# In[7]:


# Read in test data
X_test, y_test, images_test, labels_test = etl(testPath)

