import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d

# -- FORMAT DATASET --
data = open("data","r").read().split("\n")
dataset_id_one = data[0]
data = data[1:]
data = data[:-1]

index = 0
for row in data:
    if row[0] == "#":
        break
    index += 1

data_one = data[:index]
data_two = data[index:]
dataset_id_two = data_two[0]
data_two = data_two[1:]

print("DATASET ONE ID: {}".format(dataset_id_one))
print("DATASET TWO ID: {}".format(dataset_id_two))


data_one = [x.split(",") for x in data_one]
data_two = [x.split(",") for x in data_two]

f1_one = []
f2_one = []
f1_two = []
f2_two = []
features_one = []
features_two = []
label_one = []
label_two = []

for (row_one, row_two) in zip(data_one,data_two):
    f1_one.append(float(row_one[0]))
    f2_one.append(float(row_one[1]))
    features_one.append([float(row_one[0]), float(row_one[1])])
    label_one.append(float(row_one[2]))

    f1_two.append(float(row_two[0]))
    f2_two.append(float(row_two[1]))
    features_two.append([float(row_two[0]), float(row_two[1])])
    label_two.append(float(row_two[2]))

# -- END OF FORMATTING

# --- FIRST DATASET ---

# -- QUESTION ONE --
# a
C_values = [0.01, 0.1, 1, 10, 100]
poly_orders = [1,2,3,4,5,6]

kf = KFold(n_splits=5)

for train, test in kf.split(features_one):
    for c in C_values:
        for p in poly_orders:
            x_poly = PolynomialFeatures(p).fit_transform(np.array(features_one)[train])
            model = LogisticRegression(penalty='l2',C=c, max_iter=500).fit(x_poly,np.array(label_one)[train])
            