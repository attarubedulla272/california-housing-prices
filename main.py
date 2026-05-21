import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


# load the dataset
load_data=pd.read_csv("data/housing.csv")
# print(load_data.head())
# check for null values

load_data["income_cat"]=pd.cut(load_data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

# print(load_data.head())

# split the data into training and testing sets
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(load_data,load_data["income_cat"]):
    train=load_data.iloc[train_index].drop("income_cat",axis=1)
    test=load_data.iloc[test_index].drop("income_cat", axis=1)
    print(train.shape)
    print(test.shape)