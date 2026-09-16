import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

pima_df = pd.read_csv('diabetes.csv')


#print(pima_df.head())

#print(pima_df.describe())

#print(pima_df.info())

shape = pima_df.shape
print(shape)

#print(pima_df.isnull().sum())


# Features
X = pima_df.iloc[:, :-1].values
y = pima_df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

clf = DecisionTreeClassifier(
        criterion='entropy',
        random_state=100,
        max_depth=3,
        min_samples_leaf=5
    )

clf = clf.fit(X_train, y_train)


dtree_y_pred = clf.predict(X_test)

dtree_y_true = y_test

accuracy = metrics.accuracy_score(dtree_y_true, dtree_y_pred)
precision = metrics.precision_score(dtree_y_true, dtree_y_pred, average='weighted')
recall = metrics.recall_score(dtree_y_true, dtree_y_pred, average='weighted')
f1_score = metrics.f1_score(dtree_y_true, dtree_y_pred, average='weighted')

print("Accuracy: ", accuracy)
print("Precision Score: ", precision)
print("Recall Score: ", recall)
print("F1 Score: ", f1_score)

from sklearn.metrics import roc_curve, auc

dtree_auc=0

def plot_roc(dt_y_true, dt_probs):
    dtree_fpr, dtree_tpr, threshold = roc_curve(dt_y_true, dt_probs)
    dtree_auc_val = auc(dtree_fpr, dtree_tpr)
    print('AUC=%0.2f'%dtree_auc_val)

    plt.plot(dtree_fpr, dtree_tpr, label='AUC=%0.2f'%dtree_auc_val, color='darkorange')
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'b--')
    plt.xlim([0,1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return dtree_auc_val

dtree_probs = clf.predict_proba(X_test) [:,1]
dtree_auc = plot_roc(dtree_y_true, dtree_probs)


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

max_acc, max_k = 0, 0

for k in range(2, 11):
    skfold = StratifiedKFold(n_splits=k, random_state=100, shuffle=True)

    results_skfold_acc= (cross_val_score(clf, X, y, cv=skfold)).mean() * 100.0

    if results_skfold_acc > max_acc:
        max_acc = results_skfold_acc
        max_k = k

    print("Accuracy: %.2f%%" % (results_skfold_acc))

best_accuracy= max_acc
best_k_fold= max_k

print(best_accuracy, best_k_fold)
