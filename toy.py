import faulthandler

faulthandler.enable()

import cmake_example
import numpy as np

import pandas as pd 
data = pd.read_csv("/root/cmake_example/boston.csv")
trainX = data.iloc[:100, :-1].astype("double")
testX = data.iloc[100:120, :-1].astype("double")


trainY =  data.iloc[:100, -1].astype("double")
testY = data.iloc[100:120, -1].astype("double")
print("trainY mean", trainY.mean())
print("testY mean", testY.mean())

# trainX = np.array([[10,12],[9,8],[3,7],[-1,3],[12,13],[0,0],[9,9]]).astype("double")

# testX = np.array([[12,12],[7,8]]).astype("double")

# trainY = np.array([11, 8.5, 4, 0, 14, -1, 11]).astype("double")
# testY = np.array([11,8]).astype("double")


fit = cmake_example.pythonRegWithGivenXYReturnList(trainX, trainY, testX, testY, 10)
print("variable Importance:", fit.getVarImp())
print("prediction:", fit.getPrediction())
print("OOB prediction:", fit.getOOBPrediction())
# print(fit)
# print(fit.Prediction)
pred = cmake_example.pythonRegPrediction(testX, fit)
print("Result on Test Data in Python", pred.getTestPrediction())
# print(result)
# print(pred.__array_interface__['data'])

