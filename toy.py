import faulthandler

faulthandler.enable()

import cmake_example
import numpy as np


trainX = np.array([[10,12],[9,8],[3,7],[-1,3],[12,13],[0,0],[9,9]]).astype("double")

testX = np.array([[12,12],[7,8]]).astype("double")

trainY = np.array([11, 8.5, 4, 0, 14, -1, 11]).astype("double")
testY = np.array([11,8]).astype("double")

result = cmake_example.pythonRegWithGivenXY(trainX, trainY, testX, testY, 2)
print(result)
