import tensorflow as tf
import pandas as pd
print(pd.__version__)
print(tf.__version__)

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())
california_housing_dataframe.hist('housing_median_age')