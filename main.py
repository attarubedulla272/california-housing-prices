# load of dataset to predict the housing prices in California
from ast import main
import email
from unicodedata import name

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


# load the dataset
load_data=pd.read_csv("data/housing.csv")
print(load_data.head())


