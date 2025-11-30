import numpy as np
np1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np1[:,1:2])

#arithmetic
np2= np.array([1,2,3])
print (np2+34)
#"broadcasting"
#means multiplication of matrix rules got it
np3= np.array([[1,2,3,4,]])
np4= np.array([[1],[2],[3],[4]])
#1x4 and 4x1 so mutliplication gives 4x4
print (np3*np4)


# aggerate functions 
np5= np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(np.sum(np5))
print(np.mean(np5))
print(np.std(np5))
print(np.var(np5))

#filtering
np6= np.array([[67,20,30,19,5],[16,27,68,19,10]])

teenagers= np6 [ np6 < 18]
adults= np6 [ np6 >= 18]
even= np6[np6%2==0]
print(teenagers , adults , even)

#preserving the array after filtering 
adults_1= np.where(np6>=18 , np6 ,0 )  #replacing with 0
print(adults_1)

#random numbers
#randomnumbergenerator
rng= np.random.default_rng()
print(rng.integers(1,7))
#1,7 is the range or
print(rng.integers(low=1,high=7))
#for arrays 

print(rng.integers(low=1,high=101, size=6))
print(rng.integers(low=1,high=7,size=(2,3)))

fruits= np.array(["apple","banana","orange","cocnut","pineapple"])
fruits= rng.choice(fruits,size=(3,3))

print(fruits)


#pandas practice 
import pandas as pd 
data = [100,200,300,400,600]
#series is for 1d
series = pd.Series(data, index=["a","b","c","d","e"])
print(series[series>=200])
#data frame is for 2d

data1= {"Name": ["manish","ayush","ram","rahul"], "AGE":
    [18,20,30,90]}
df= pd.DataFrame(data1)

print(df.loc[1])
#new column addition 
df["job"] = ["w","e","y","y"]


#new row add krna
row1=pd.DataFrame([{"Name" :"sandy","AGE":28,"job":"t"}])#all attributes must be same like Nmae , AGE, job
df= pd.concat([df,row1])
print(df)

# simple linear regression practice 
import numpy as np


x = np.array([1, 2, 3], dtype=float)
y = np.array([2, 4, 6], dtype=float)

m = 0
c = 0
lr = 0.1


for i in range(1000):
    y_pred = m * x + c
    m = m - lr * np.mean((y_pred - y) * x)
    c = c - lr * np.mean(y_pred - y)

print("m =", m)
print("c =", c)

#normal equation sampling 
import numpy as np
import matplotlib.pyplot as plt 

x = np.array([1, 2, 3,6,89,45,67], dtype=float)
y = np.array([2, 4, 6,9,10,53,76], dtype=float)
x=x.reshape(-1,1)
x_bias = np.c_[np.ones((x.shape[0], 1)), x]
XtX = x_bias.T.dot(x_bias)
theta = np.linalg.inv(XtX).dot(x_bias.T).dot(y)
print(theta)
c = theta[0]
m = theta[1]