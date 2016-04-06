from sklearn import svm
import pandas as pn
train_data = pn.read_csv("svm-data.csv",header=None)
train_data_class = train_data[:][0]
train_data_params = train_data
train_data_params.drop(train_data_params.columns[[0]],axis = 1, inplace = True)

#print(train_data_params,train_data_class)

svc = svm.SVC(kernel='linear', C=100000, random_state=241)
svc.fit(train_data_params,train_data_class)
#print(svc.support_)
a=svc.coef_[0]
print(a[0],a[1])
