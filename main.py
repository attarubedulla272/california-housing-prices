import os
import numpy as np
import pandas as pd
import joblib 

import matplotlib.pyplot as plt
from sklearn import pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes):
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("standard_scaler",StandardScaler())
    ])

    cat_pipeline=Pipeline([
        ("one_hot_encoder",OneHotEncoder())
    ])

    full_pipeline=ColumnTransformer([
        ("numerical",num_pipeline,num_attributes),
        ("categorical",cat_pipeline,cat_attributes)
    ])

    return full_pipeline

# if os.path.exists(MODEL_FILE):
#     print("Model already exists. Skipping training.")

data=pd.read_csv("./data/housing.csv")

data['income_cat']=pd.cut(data['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in split.split(data,data['income_cat']):
    strat_train_set=data.loc[train_index].drop(columns=['income_cat'])
    strat_test_set=data.loc[test_index].drop(columns=['income_cat'])


housing=strat_train_set.copy()



housing_labels=housing['median_house_value'].copy()
housing=housing.drop(columns=['median_house_value'])



num_attributes=housing.drop("ocean_proximity",axis=1).columns
cat_attributes=["ocean_proximity"]



full_pipeline=build_pipeline(num_attributes,cat_attributes)
housing_prepared=full_pipeline.fit_transform(housing)



