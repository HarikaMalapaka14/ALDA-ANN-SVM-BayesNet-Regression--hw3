import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
data=pd.read_csv('hw3q3(b).csv')
df=pd.DataFrame(data)



"""

mod1 = smf.ols(formula='y ~ x1 + x2 + x3 + x4', data=df)
res1 = mod1.fit()
print("model 1")
print(res1.params)



mod2 = smf.ols(formula='y ~ np.power(x1,2) + np.power(x2,2) + np.power(x3,2) + np.power(x4,2)', data=df)
res2 = mod2.fit()
print("model 2")
print(res2.params)


mod3 = smf.ols(formula='y ~ np.power(x1,3) + np.power(x2,3) + np.power(x3,3) + np.power(x4,3)', data=df)
res3 = mod3.fit()
print("model 3")
print(res3.params)

"""
output1=[]
output2=[]
output3=[]


######### part ii)

for ind,i in enumerate(df.values):
	df_test=pd.DataFrame(df.iloc[[ind],])
	df_train=df.drop(df.index[ind])
	df_test.columns=['x1','x2','x3','x4','y']
	df_train.columns=['x1','x2','x3','x4','y']
	model1='model1{}'.format(ind)
	model2='model2{}'.format(ind)
	model3='model3{}'.format(ind)
	locals()[model1]=smf.ols(formula='y ~ x1 + x2 + x3 + x4', data=df_train).fit()
	para1=locals()[model1].params
	ans1=((para1[1]*df_test['x1'])+(para1[2]*df_test['x2'])+(para1[3]*df_test['x3'])+(para1[4]*df_test['x4'])+(para1[0]))

	output1.append(float(ans1))
	
	locals()[model2]=smf.ols(formula='y ~ np.power(x1,2) + np.power(x2,2) + np.power(x3,2) + np.power(x4,2)', data=df_train).fit()
	para2=locals()[model2].params
	ans2=((para2[1]*np.power(df_test['x1'],2))+(para2[2]*np.power(df_test['x2'],2))+(para2[3]*np.power(df_test['x3'],2))+(para2[4]*np.power(df_test['x4'],2))+(para2[0]))
	output2.append(float(ans2))

	locals()[model3]=smf.ols(formula='y ~ np.power(x1,3) + np.power(x2,3) + np.power(x3,3) + np.power(x4,3)', data=df_train).fit()
	para3=locals()[model3].params
	ans3=((para3[1]*np.power(df_test['x1'],3))+(para3[2]*np.power(df_test['x2'],3))+(para3[3]*np.power(df_test['x3'],3))+(para3[4]*np.power(df_test['x4'],3))+(para3[0]))
	output3.append(float(ans3))



actuals=np.array(df['y'])
errors1=[]
errors2=[]
errors3=[]


for i in range(0,1000):
	diff1=sqrt(np.power((actuals[i]-output1[i]),2))
	errors1.append(diff1)


	diff2=sqrt(np.power((actuals[i]-output2[i]),2))
	errors2.append(diff2)

	diff3=sqrt(np.power((actuals[i]-output3[i]),2))
	errors3.append(diff3)





print(np.mean(errors1))
print(np.mean(errors2))
print(np.mean(errors3))



from sklearn.metrics import mean_squared_error

o1=np.array(output1)
o2=np.array(output2)
o3=np.array(output3)
rmse1=np.sqrt(np.square(np.subtract(actuals,o1)).mean())
rmse2=np.sqrt(np.square(np.subtract(actuals,o2)).mean())
rmse3=np.sqrt(np.square(np.subtract(actuals,o3)).mean())

print(rmse1, rmse2, rmse3)




	
