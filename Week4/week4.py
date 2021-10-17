from re import X
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d
import json

data = open("data","r").read().split("\n")
dataset_id = data[0]
data = data[1:]
data = data[:-1]
print("DATASET ID: {}".format(dataset_id))
data = [a.split(",") for a in data]