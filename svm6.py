from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
data=pd.read_csv('hw3q6.csv')
df=pd.DataFrame(data)

df_Y=df.iloc[:,[-1]]
df_X=df.loc[:, df.columns != 'Class']

class_pos=0
class_neg=0

for i in df_Y.values:
	if(i[0]==1):
		class_pos+=1
	else:
		class_neg+=1


print(class_pos, class_neg)
X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.25, stratify=df_Y, random_state=8)

cls_pos_train=0
cls_neg_train=0
cls_pos_test=0
cls_neg_test=0

for i in y_train.values:
	if(i[0]==1):
		cls_pos_train+=1
	else:
		cls_neg_train+=1

for i in y_test.values:
	if(i[0]==1):
		cls_pos_test+=1
	else:
		cls_neg_test+=1


print("train set: positive", cls_pos_train)
print("train set : negative", cls_neg_train)
print("test set : positive", cls_pos_test)
print("test set : negative", cls_neg_test)

y_tr=np.array(y_train['Class'])


num_sv=[]
c= [0.1, 0.5, 1, 5, 10, 50, 100]
for i in c:
	clf=svm.SVC(C=i,gamma='auto')
	clf.fit(X_train,y_tr)
	num_sv.append(clf.n_support_)


for i in num_sv:
	print(i, sum(i))





parameters = {'C':[0.1,0.5,1,5,10,50,100],'gamma':[0.1,2,5,10], 'degree':[2,3,4,5],'coef0':[0,1,2.5, 4,5.5]}

clf_grid_linear=svm.SVC(kernel='linear')
grid_linear = GridSearchCV(clf_grid_linear, parameters, cv=5)
grid_linear.fit(X_train,y_tr)


clf_grid_rbf=svm.SVC(kernel='rbf')
grid_rbf = GridSearchCV(clf_grid_rbf, parameters, cv=5)
grid_rbf.fit(X_train,y_tr)


clf_grid_poly=svm.SVC(kernel='poly')
grid_poly = GridSearchCV(clf_grid_poly, parameters, cv=5)
grid_poly.fit(X_train,y_tr)


clf_grid_sigmoid=svm.SVC(kernel='sigmoid')
grid_sigmoid = GridSearchCV(clf_grid_sigmoid, parameters, cv=5)
grid_sigmoid.fit(X_train,y_tr)

print("Linear")
print(grid_linear.best_score_)
print(grid_linear.best_estimator_)

print("RBF")
print(grid_rbf.best_score_)
print(grid_rbf.best_estimator_)

print("Poly")
print(grid_poly.best_score_)
print(grid_poly.best_estimator_)

print("Sigmoid")
print(grid_sigmoid.best_score_)
print(grid_sigmoid.best_estimator_)




table=pd.DataFrame(columns=['kernal','Accuracy','Precision','Recall','F-measure'])
print("reached")

y_te=np.array(y_test['Class'])

at_pos=0
for i in ['linear', 'rbf','poly','sigmoid']:
	at_pos+=1
	if(i=='linear'):
		algo=grid_linear.best_estimator_
	if(i=='rbf'):
		algo=grid_rbf.best_estimator_
	if(i=='poly'):
		algo=grid_poly.best_estimator_
	else:
		algo=grid_sigmoid.best_estimator_

	pred=algo.predict(X_test)
	accu=accuracy_score(y_te,pred)
	rec=recall_score(y_te,pred)
	prec=precision_score(y_te,pred)
	f1=f1_score(y_te,pred)
	table.at[at_pos,'kernal']=i
	table.at[at_pos,'Accuracy']=accu
	table.at[at_pos,'Precision']=prec
	table.at[at_pos,'Recall']=rec
	table.at[at_pos,'F-measure']=f1




print(table)
