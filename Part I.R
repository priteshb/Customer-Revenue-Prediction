library(readr)
library(Amelia)
library(ggplot2)
library(mlbench)
library(robustbase)
library(car)
library(clusterSim)
library(EnvStats)
library(MASS)
library(VIM)
library(ForImp)
library(ggbiplot)
library(mice)
library(randomForest)
library(caret)
library(corrplot)
library(dplyr)
library(corrr)
library(imputeMissings)
library(doParallel)
library(foreach)
library(e1071)
library(Hmisc)
library(parallelSVM)

data = read.csv("C:/OU/DSA/Fall 2018/Predictathon/I/Train.csv", header = TRUE)
data_test = read.csv("C:/OU/DSA/Fall 2018/Predictathon/I/Test.csv", header = TRUE)
sessionId = data$sessionId
sessionId_test = data_test$sessionId
data = subset(data, select = -c(custId,sessionId,date,year,month))
data_test = subset(data_test, select = -c(custId,sessionId,date,year,month))

data[data == ""] = "NA"
data_test[data_test == ""] = "NA"
mv = colnames(data)[colSums(is.na(data)) > 0] #Columns having missing values
sapply(data[mv],function(x)mean(is.na(x)))
# missmap(data)
col_rem = which(colMeans(is.na(data)) > 0.2)
data = data[,-col_rem]
data_test = data_test[,-col_rem]

mv = colnames(data)[colSums(is.na(data)) > 0] #Columns having missing values
# aggr(data, numbers = TRUE,labels = names(data), cex.axis = .7, ylab = c("Histogram of missing data","Missingness Pattern"))
# data = kNN(data, variable = mv, k = 10, imp_var = FALSE)
# data = impute(data, object = NULL, method = "median/mode", flag = FALSE)
# data_test = impute(data_test, object = NULL, method = "median/mode", flag = FALSE)
data$medium = impute(data$medium, median)
data$operatingSystem=impute(data$operatingSystem, median)
data$continent = impute(data$continent, median)
data$subContinent = impute(data$subContinent, median)
data$country = impute(data$country, median)
data$pageviews = impute(data$pageviews, median)
data$source = impute(data$source, median)
data$browser = impute(data$browser, median)

# impute test data
data_test$medium = impute(data_test$medium, median)
data_test$operatingSystem=impute(data_test$operatingSystem, median)
data_test$continent = impute(data_test$continent, median)
data_test$subContinent = impute(data_test$subContinent, median)
data_test$country = impute(data_test$country, median)
data_test$pageviews = impute(data_test$pageviews, median)
data_test$source = impute(data_test$source, median)
data_test$browser = impute(data_test$browser, median)

nu = unlist(lapply(data, is.numeric))
numCols = names(data)[which(nu == "TRUE")] 
facCols = names(data)[which(nu == "FALSE")] 

nu_test = unlist(lapply(data_test, is.numeric))
numCols_test = names(data_test)[which(nu_test == "TRUE")]
facCols_test = names(data_test)[which(nu_test == "FALSE")]

data.factors = data[facCols]
data.numeric = data[numCols]

data_test.factors = data_test[facCols_test]
data_test.numeric = data_test[numCols_test]

irr_facs = aov(data$revenue~., data = data.factors)
summary(irr_facs)
data = subset(data, select = -c(deviceCategory, country, source))
data.factors = subset(data.factors, select = -c(deviceCategory, country, source))
data_test = subset(data_test, select = -c(deviceCategory, country, source))
data_test.factors = subset(data_test.factors, select = -c(deviceCategory, country, source))

data.factors.dv = dummyVars( ~ ., data = data.factors, fullRank = TRUE)
data.factors.dv = predict(data.factors.dv, newdata = data.factors)
data_test.factors.dv = dummyVars( ~ ., data = data_test.factors, fullRank = TRUE)
data_test.factors.dv = predict(data_test.factors.dv, newdata = data_test.factors)
cn_test = colnames(data_test.factors.dv)
data.factors.dv = data.factors.dv[,c(cn_test)]
cn = colnames(data.factors.dv)
data_test.factors.dv = data_test.factors.dv[,c(cn)]

data = data.frame(cbind(data.factors.dv,data.numeric))
data1 = subset(data, select = -c(revenue))
nzv = nearZeroVar(data1)
data1 = data1[,-nzv]
data_test = data.frame(cbind(data_test.factors.dv,data_test.numeric))
data1_test = data_test[,-nzv]

corMat = cor(data1)
f = findCorrelation(corMat,cutoff = 0.8)
data1 = data1[,-f]
data1_test = data1_test[,-f]

# corrplot(cor(data1[,unlist(lapply(data1, is.numeric))][-1]), 
         # type = 'upper', order = "AOE")

data$positiveTransaction = ifelse(data$revenue >0, 1, 0)

data_new = data.frame(cbind(data1,data$positiveTransaction))
names(data_new)[names(data_new) == "data.positiveTransaction"] = "positiveTransaction"
data_new$positiveTransaction = as.factor(data_new$positiveTransaction)

modelFit = foreach(ntree=1000, .combine=randomForest::combine, .multicombine=TRUE,
              .packages='randomForest') %dopar% {
                randomForest(positiveTransaction ~., data = data_new, ntree=ntree)
              }

prediction = predict(modelFit, data1_test)
# cm = as.matrix(table(Actual = testingSet$positiveTransaction, Predicted = prediction))
mpred = read.csv("C:/OU/DSA/Fall 2018/Predictathon/I/Book1.csv",header = TRUE)
output = data.frame(sessionId_test,mpred)
names(output)[names(output) == "sessionId_test"] = "sessionId"
names(output)[names(output) == "mpred"] = "predictedClass"
options(scipen=999)
write.csv(output,file = "C:/OU/DSA/Fall 2018/Predictathon/I/Submission_rf_2.csv", row.names = FALSE)

modelsvm = svm(positiveTransaction ~., data = data_new, numberCores = detectCores(),
                       scale = TRUE, kernel = "radial", degree = 3, 
                       gamma = 0.1)

prediction = predict(modelsvm, data1_test)

p = cm[1]/(cm[1]+cm[3])
r = cm[1]/(cm[1]+cm[2])
F1 = 2*p*r/(p+r)
