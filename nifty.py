import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
df=pd.read_csv("nifty data1.csv")
#print(df.head())
x=df.drop(['Close','Date'],axis=1)
#x=df.drop('Date',axis=1)
y=df['Close']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
#print(x.head())
from sklearn import linear_model
lasso=linear_model.Lasso(alpha=0.1,tol=9.126e+05,max_iter=1000,).fit(x_train,y_train)




pickle.dump(lasso, open('nifty.pkl', 'wb'))