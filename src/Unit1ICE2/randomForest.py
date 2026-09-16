# loads the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# load the iris dataset
iris = pd.read_csv('iris.csv')
iris.columns = iris.columns.str.strip()

x = iris[['petal_length', 'sepal_length']]

y = iris[['species']]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# standardize x_train and x_test
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

cart = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_leaf=5)
cart.fit(x_train, y_train)

y_pred = cart.predict(x_test)

conf = confusion_matrix(y_test, y_pred)

print(conf)