
import numpy as np
import pandas as pd
from sklearn import linear_model
from google.colab import files
uploaded = files.upload()

"""### Import the data and remove useless columns"""

df = pd.read_csv("train.csv")
df.drop(columns=["Id"],inplace=True)
df.head()


"""### Replace categorical data (strings) with numerical values"""

obj_to_replace = df["MSZoning"].dtype

for column in df.columns:
    if df[column].dtype == obj_to_replace:
        uniques = np.unique(df[column].values)
        for idx,item in enumerate(uniques):
            df[column] = df[column].replace(item,idx)
            
df.head()

"""### Handle the missing data (NaNs)"""
df.drop(columns=df.columns[df.isnull().sum().values>200],inplace=True)
df.dropna(inplace=True)
df.isnull().sum().values

"""### Add the bias column (column of ones)"""

df["bias"] = np.ones(df.shape[0])
df.head()

"""### Train the linear regressor"""

# Create and fit the model
reg = linear_model.LinearRegression()
reg.fit(df[['MSSubClass',	'MSZoning',	'LotArea',	'Street',	'LotShape',	'LandContour',	'Utilities',	'LotConfig',	'LandSlope',	'Neighborhood',	'Condition1',	'Condition2',	'BldgType',	'HouseStyle',	'OverallQual',	'OverallCond',	'YearBuilt',	'YearRemodAdd',	'RoofStyle',	'RoofMatl',	'Exterior1st',	'Exterior2nd',	'MasVnrType',	'MasVnrArea',	'ExterQual',	'ExterCond',	'Foundation',	'BsmtQual',	'BsmtCond',	'BsmtExposure',	'BsmtFinType1',	'BsmtFinSF1',	'BsmtFinType2',	'BsmtFinSF2',	'BsmtUnfSF',	'TotalBsmtSF',	'Heating',	'HeatingQC',	'CentralAir', 'Electrical',	'1stFlrSF',	'2ndFlrSF',	'LowQualFinSF',	'GrLivArea',	'BsmtFullBath',	'BsmtHalfBath',	'FullBath',	'HalfBath',	'BedroomAbvGr',	'KitchenAbvGr',	'KitchenQual',	'TotRmsAbvGrd',	'Functional',	'Fireplaces',	'GarageType',	'GarageYrBlt',	'GarageFinish',	'GarageCars',	'GarageArea',	'GarageQual', 'GarageCond',	'PavedDrive',	'WoodDeckSF',	'OpenPorchSF',	'EnclosedPorch',	'3SsnPorch',	'ScreenPorch',	'PoolArea',	'MiscVal',	'MoSold',	'YrSold',	'SaleType',	'SaleCondition']],df.SalePrice)
reg.coef_


from sklearn.linear_model import LinearRegression
reg.intercept_

test_file = files.upload()

tf = pd.read_csv("test.csv")
tf.drop(columns=["Id"],inplace=True)
tf.head()

####getting tested data#########

obj_to_replace = tf["MSZoning"].dtype

for column in tf.columns:
    if tf[column].dtype == obj_to_replace:
        uniqued = np.unique(tf[column].values)
        for idx,item in enumerate(uniqued):
            tf[column] = tf[column].replace(item,idx)
            
tf.head()
######predict the desired house price
reg.predict([[]])
