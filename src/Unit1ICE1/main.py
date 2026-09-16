import pandas as pd


cars_df =  pd.read_csv('Cars.csv')

userNum = 5

# Subset the first userNum rows of the data frame
# and select the maximum elements of each field
max_df = cars_df.iloc[:userNum].max()
