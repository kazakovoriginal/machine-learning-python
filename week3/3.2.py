import sklearn
from sklearn import svm
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

#********** Обработка данных**********
newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
vectorizer = TfidfVectorizer()
#********** Нормализация данных**********
data_params = vectorizer.fit_transform(newsgroups.data)
#print(data_params)

#********** Доступ к именам**********
feature_mapping = vectorizer.get_feature_names()
#print(feature_mapping)
"""

#********** Нахождение такого значения С, пр котором точность максимальна**********
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(newsgroups.target.size, n_folds=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = sklearn.grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(data_params, newsgroups.target)

#print(gs.grid_scores_)
"""
#********** Создание и обучение классификатора с оптимальным значением С**********
svc = svm.SVC(kernel='linear', C=1, random_state=241)
svc.fit(data_params, newsgroups.target)

#********** Получение веса каждого слова и преобразование в формат, с которым можно работать **********
myList=[]
myList = svc.coef_
a1=np.hstack(myList)
a2=a1[0].toarray()
a3=a2[0]
d=dict([(a3[0],0)])

#********** Создание словаря: key = вес; value = id слова**********
for i in range(0,len(a3)):
    d[abs(a3[i])]=i
'''
#********** Сортировка словаря и получение 10 лучших весов и id слов для этих весов **********
print(d)

keylist = d.keys()
keylist1=sorted(keylist)
for key in keylist1:
    print("%s: %s" % (key, d[key]))
'''
index = [22936,15606,5776,21850,23673,17802,5093,5088,12871,24019]


#********** Создание и сортировка спсика из нужных 10 слов**********
list=[]
for i in index:
    list.append(feature_mapping[i])
list.sort()
str = ""

#********** Вывод из в строку через пробел**********
for word in list:
    str+=word + " "
print(str)


#print(svc.coef_.shape)
#print(sorted(map(abs, res)))

#for a in gs.grid_scores_:
#    print (a.mean_validation_score,a.parameters)
