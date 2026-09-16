import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_csv('realestate.csv')
df.head()

shape_data = df.shape
print(shape_data)

df.info()

df.describe()

X2_max = df['X2 house age'].max()
X4_max = df['X4 number of convenience stores'].max()
print(X2_max)
print(X4_max)

X3_null = df['X3 distance to the nearest MRT station'].isnull().sum()
X4_null = df['X4 number of convenience stores'].isnull().sum()
Y_null = df['Y house price of unit area'].isnull().sum()
print(X3_null)
print(X4_null)
print(Y_null)

df_drop = df.copy()
df_drop = df_drop.dropna(axis = 0)

mean_X3_drop = df_drop['X3 distance to the nearest MRT station'].mean()
print(mean_X3_drop)

df_fill = df.copy()
df_fill = df_fill.fillna(df_fill.median())

mean_X3_fill = df_fill['X3 distance to the nearest MRT station'].mean()
print(mean_X3_fill)

dataframe = df_fill.copy()

dataframe = dataframe[
        (dataframe['Y house price of unit area'] <= 80) &
        (dataframe['X3 distance to the nearest MRT station'] <= 2800) &
        (dataframe['X6 longitude'] >= 121.5)
    ]

dataframe['Y house price of unit area'] = dataframe['Y house price of unit area'].apply(lambda x: x * 91)

from sklearn import preprocessing

normalized = preprocessing.MinMaxScaler().fit_transform(dataframe)

normalized_df = pd.DataFrame(normalized, columns=dataframe.columns)

mean_norm_Y = normalized_df['Y house price of unit area'].mean()
print(mean_norm_Y)

def age_classifier(age):
    if age < 10:
        return 'New'
    elif age < 30:
        return 'Middle'
    else:
        return 'Old'

df_age_classify = dataframe.copy()

df_age_classify['Age Class'] = df_age_classify['X2 house age'].apply(age_classifier)

New_count = (df_age_classify['Age Class'] == 'New').sum()
Middle_count = (df_age_classify['Age Class'] == 'Middle').sum()
Old_count = (df_age_classify['Age Class'] == 'Old').sum()
print(New_count)
print(Middle_count)
print(Old_count)

reindex_df = dataframe.set_index('No')
max_id = reindex_df['Y house price of unit area'].idxmax()

print(max_id)

txn_dt = reindex_df.loc[max_id, 'X1 transaction date']
house_age = reindex_df.loc[max_id, 'X2 house age']
conv_st = reindex_df.loc[max_id, 'X4 number of convenience stores']
print(txn_dt)
print(house_age)
print(conv_st)

age_price_df = reindex_df[
        (reindex_df['X2 house age'] <= 9) &
        (reindex_df['Y house price of unit area'] > 27)
    ]

grouped_age_price_df = age_price_df.groupby('X4 number of convenience stores')['Y house price of unit area'].mean()

mean_val_conv_7 = round(grouped_age_price_df.loc[7], 2)
print(mean_val_conv_7)

grouped_age_price_df.plot.bar(
            title='Mean House Price based on Convenience store proximity',
            ylabel='Price of unit area',
            xlabel='Number of convenience stores',
            figsize=(6,5)
    )
plt.show()