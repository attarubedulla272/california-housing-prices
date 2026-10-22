import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor



def bulidpipeline(num_attributes,cat_attributes):
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("standard_scaler",StandardScaler())
    ])
    cat_pipeline=Pipeline([
        ("onehotencode",OneHotEncoder())
    ])

    full_pipeline=ColumnTransformer([
        ("num_att",num_pipeline,num_attributes),
        ("cat_att",cat_pipeline,cat_attributes)
    ])

    return full_pipeline
















# load the dataset
load_data=pd.read_csv("data/housing.csv")


load_data["income_cat"]=pd.cut(load_data["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(load_data,load_data["income_cat"]):
    train=load_data.iloc[train_index].drop("income_cat",axis=1)
    test=load_data.iloc[test_index].drop("income_cat", axis=1)

   
housing_labels=train["median_house_value"].copy()
housing=train.drop("median_house_value", axis=1).copy()

num_attributes=housing.drop("ocean_proximity",axis=1).columns
cat_attributes=["ocean_proximity"]

fullpipeline=bulidpipeline(num_attributes,cat_attributes)

housing_prepared=fullpipeline.fit_transform(housing)

print(housing_prepared)



