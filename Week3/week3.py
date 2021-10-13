import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d


data = open("data","r").read().split("\n")
dataset_id = data[0]
data = data[1:]
data = data[:-1]
print("DATASET ID: {}".format(dataset_id))
data = [a.split(",") for a in data]

# -- FORMAT DATA FOR FUTURE USE
f1 = []
f2 = []
features = []
target_outputs = []

for row in data:
    f1.append(float(row[0]))
    f2.append(float(row[1]))
    target_outputs.append(float(row[2]))
    features.append([float(row[0]),float(row[1])])
train_features,test_features,train_output,test_output = train_test_split(features,target_outputs,test_size=0.25,random_state=0)

# --

# --- QUESTION ONE ---
#  -- a --
# Create 3D scatter plot of features on two axes with z axis being the target output of feature vector.

# Creating figure
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

# Creating colormap
colormap = plt.get_cmap("hsv")
 
# Creating plot
scatter_3d = ax.scatter3D(f1, f2, target_outputs, c=(target_outputs), cmap=colormap)
fig.colorbar(scatter_3d, ax = ax, shrink = 0.5, aspect = 5)
ax.set_xlabel('First feature', fontweight ='bold')
ax.set_ylabel('Second Feature', fontweight ='bold')
ax.set_zlabel('Target Output', fontweight ='bold')
plt.title("3D Scatterplot of Dataset")
 

# show plot
plt.show()

#  -- b --
