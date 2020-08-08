import keras
from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
from tensorflow import set_random_seed
set_random_seed(7)
import numpy as np
np.random.seed(7)
#tf.set_random_seed(seed)
from sklearn.metrics import accuracy_score


df1=pd.read_csv('X_train.csv')
df2=pd.read_csv('X_test.csv')
df3=pd.read_csv('X_val.csv')
df4=pd.read_csv('Y_train.csv')
df5=pd.read_csv('Y_test.csv')
df6=pd.read_csv('Y_val.csv')

X_train=pd.DataFrame(df1)
X_test=pd.DataFrame(df2)
X_val=pd.DataFrame(df3)
Y_train=pd.DataFrame(df4)
Y_test=pd.DataFrame(df5)
Y_val=pd.DataFrame(df6)

accu_train=[]

accu_val=[]
for i in (2,4,6,8,10):
	clf='classifier{}'.format(i)
	locals()[clf]=Sequential()
	locals()[clf].add(Dense(output_dim = i, init = 'uniform', activation = 'relu', input_dim = 61))
	locals()[clf].add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
	locals()[clf].compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
	locals()[clf].fit(X_train,Y_train, batch_size = 50, epochs = 10)
	pred_train=np.round(locals()[clf].predict(X_train))
	print("Training set")
	print(pred_train)
	pred_val=np.round(locals()[clf].predict(X_val))
	print("Validation set")
	acc_train=accuracy_score(Y_train,pred_train)
	acc_val=accuracy_score(Y_val,pred_val)
	accu_train.append(acc_train)
	accu_val.append(acc_val)
	


## p=4 is best
## on test set using 6 hidden layers


clf=Sequential()
clf.add(Dense(output_dim=10,init='uniform',activation='relu', input_dim=61))

clf.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
clf.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

clf.fit(X_train,Y_train, batch_size = 50, epochs = 10)
pred_test=np.round(clf.predict(X_test))
test_accuracy=accuracy_score(pred_test, Y_test)


print("Accuracy train ",accu_train)
print("Accuracy validate",accu_val)

print("The accuracy of test is ", test_accuracy)





