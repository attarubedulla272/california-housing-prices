# load of dataset to predict the housing prices in California
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# load the dataset
load_data=pd.read_csv("data/housing.csv")
print(load_data.head())
# check for null values
