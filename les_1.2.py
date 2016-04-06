import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn import tree
import pandas
import numpy as np

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
clf = tree.DecisionTreeClassifier(random_state=241)
data_without_nan = data.dropna(axis=0)

# print(data_without_nan)

data_fore = np.column_stack((data_without_nan['Age'].values,data_without_nan["Fare"].values,
                            data_without_nan['Pclass'].values,data_without_nan['Sex'].values))


# print(data_fore,len(data_fore))

new_list = []
for a in data_without_nan['Sex']:
    new_list.append(int(a.replace("female","0").replace("male","1")))
# print(new_list)
data_new = np.column_stack((data_without_nan['Age'].values,data_without_nan["Fare"].values,
                            data_without_nan['Pclass'].values,pandas.DataFrame({'Sex':new_list}).values))

'''
print(len(data_without_nan.values),len(data_without_nan['Sex']),len(new_list), len(data_new))
print(data_new)
'''
data_check = np.column_stack((data_new,data_without_nan['Survived'].values))

# print(data_check)

clf.fit(data_new,data_without_nan['Survived'].values)

print(clf.feature_importances_[0],clf.feature_importances_[1],clf.feature_importances_[2],clf.feature_importances_[3])

"""
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
os.unlink('iris.dot')
"""