# loads the necessary libraries
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing  import LabelEncoder as Label_Encoder
# load the car dataset
df = pd.read_csv('car_evaluation.csv', header=None)

print(df.isnull().sum())

# Encode the data
encoder = Label_Encoder()

cols = df.columns

for x in cols:
    df[x] = encoder.fit_transform(df[x])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

rfc = RandomForestClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_leaf=5)
rfc.fit(x_train, y_train)

y_pred = rfc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)