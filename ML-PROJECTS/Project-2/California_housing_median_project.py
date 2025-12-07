import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt


data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["median_house_value"] = data.target

print(df.describe())

x=df.drop(columns=["median_house_value"]) 
y=df["median_house_value"]

#splitting my data into 80:20 set training ke liye and for testing
training_size=0.8
split_size=np.random.rand(len(x))< training_size
training_x=x[split_size]
training_y=y[split_size]

test_x=x[~split_size]
test_y=y[~split_size]

print ("total training dataset is ",len(training_x))
print ("total training target value is ",len(training_y))

# Stanadard scaling 

mean_train = training_x.mean(axis=0)
std_train = training_x.std(axis=0)

training_x = (training_x - mean_train) / std_train
test_x = (test_x - mean_train) / std_train


training_x= training_x.values

#2d array m converted like matrix jisme like (nx1 matrix )
training_y=training_y.values.reshape(-1,1)
test_x= test_x.values
test_y=test_y.values.reshape(-1,1)


# adding a row for c jisse hame intercept mil ske , like adding anew row in a matrix of 1
train_x_intercept = np.c_[np.ones((training_x.shape[0], 1)), training_x] 
test_x_intercept  = np.c_[np.ones((test_x.shape[0], 1)), test_x]

#LINEAR REGRESSION USING NORMAL EQUATION.

#c_m=(X⊤X)inverse . (X⊤y) normal equation for regression 
c_m = np.linalg.inv(train_x_intercept.T.dot(train_x_intercept)).dot(train_x_intercept.T).dot(training_y)
print("intercept is " ,c_m[0] )

print("Equation of line will be y = " + str(c_m[0,0]) 
      + " + " + str(c_m[1,0]) + "*MedInc"
      + " + " + str(c_m[2,0]) + "*HouseAge"
      + " + " + str(c_m[3,0]) + "*AveRooms"
      + " + " + str(c_m[4,0]) + "*AveBedrms"
      + " + " + str(c_m[5,0]) + "*Population"
      + " + " + str(c_m[6,0]) + "*AveOccup"
      + " + " + str(c_m[7,0]) + "*Latitude"
      + " + " + str(c_m[8,0]) + "*Longitude")


y_train_pred = train_x_intercept.dot(c_m)
y_test_pred  = test_x_intercept.dot(c_m)


MSE_Training = np.mean((training_y - y_train_pred)**2)
MSE_TESTING  = np.mean((test_y - y_test_pred)**2)


print("Train MSE is ", MSE_Training)
print("Test MSE is ", MSE_TESTING)


squared_R = np.sum((test_y - y_test_pred)**2)
squared_r_mean = np.sum((test_y - np.mean(test_y))**2)
r2 = 1 - (squared_R / squared_r_mean)
print("Test R^2 is :", r2)


#graph plotting for median income and median house value ( data waala)
med_inc_train = df.loc[split_size, "MedInc"]
house_val_train = df.loc[split_size, "median_house_value"]
plt.scatter(med_inc_train, house_val_train, alpha=0.5)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Median Income vs Median House Value")
plt.show()












