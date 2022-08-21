library(RLT)

set.seed(1)

trainn = 1000
testn = 200
n = trainn + testn
p = 400
X1 = matrix(rnorm(n*p/2), n, p/2)
X2 = matrix(as.integer(runif(n*p/2)*3), n, p/2)

X = data.frame(X1, X2)
for (j in (p/2 + 1):p) X[,j] = as.factor(X[,j])
y = 1 + X[, 2] + 2 * (X[, p/2+1] %in% c(1, 3)) + rnorm(n)
#y = 1 + rowSums(X[, 1:(p/4)]) + rowSums(data.matrix(X[, (p/2) : (p/1.5)])) + rnorm(n)
#y = 1 + X[, 1] + rnorm(n)


ntrees = 50
ncores = 2
nmin = 60
mtry = p/2
sampleprob = 0.85
rule = "best"
nsplit = ifelse(rule == "best", 0, 3)
importance = TRUE 

trainX = X[1:trainn, ]
trainY = y[1:trainn]

testX = X[1:testn + trainn, ]
testY = y[1:testn + trainn]


RLTfit <- RLT(trainX, trainY, ntrees = ntrees, ncores = ncores, 
              nmin = nmin, mtry = mtry, nsplit = nsplit,
              split.gen = rule, resample.prob = sampleprob,
              importance = importance, param.control = list("alpha" = 0),
              verbose = TRUE, seed = 1)
write.csv(x = X, file = 'train.csv')
write.csv(x= y, file = 'test.csv')

write.csv(x=RLTfit.FittedForest['VarImp'], 'VarImp.csv')
write.csv(x=RLTfit[["VarImp"]], 'VarImp.csv')
write.csv(x=RLTfit[["SplitVar"]], 'SplitVar.csv')
write.csv(x=RLTfit[["SplitValue"]], 'SplitValue.csv')
write.csv(x=RLTfit[["LeftNode"]], 'LeftNode.csv')
write.csv(x=RLTfit[["RightNode"]], 'RightNode.csv')
