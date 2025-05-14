library(caret)
library(randomForest)
library(e1071)
library(class)

## Data Loading
# Loading Homo Sapies dataset
dxFeatures_GOs_Hm <- read.csv("C:/Users/flavi/Documents/Cifasis/Proyectos/BiotrAIn/dxFeatures_GOs_Hm.csv", row.names=1)

# To see the information of characterized and GO data
dim(dxFeatures_GOs_Hm)

colnames(dxFeatures_GOs_Hm)[1:20]

rownames(dxFeatures_GOs_Hm)[1:10]

head.matrix(dxFeatures_GOs_Hm[, 466:476], n = 10)

# Checking the amount of annotations by GO-term
apply(dxFeatures_GOs_Hm[, c("GO.0006914","GO.0090398","GO.0042981", "GO.0034599")], MARGIN=2, sum)

# Binary classification
## Builing of the test and train datasets

# Choice of biological function to predict
testGOs <- c("GO.0042981")

# Transform to a vector of type factor
newClassGOHm <- as.factor(dxFeatures_GOs_Hm[, testGOs])

# Generation of test and train indices
indexData <- createFolds(t(newClassGOHm), k = 5)
indexTest <- indexData[[4]]
indexTrain <- setdiff(1:dim(dxFeatures_GOs_Hm)[1], indexTest)

# Building of the dataset to be applied to ML methods
dxHm <- dxFeatures_GOs_Hm[, !(names(dxFeatures_GOs_Hm) %in% c("GO.0006914","GO.0090398", "GO.0034599", "GO.0045087"))]
dxHm[, testGOs] <- as.factor(dxHm[, testGOs]) 

## K-Nearest Neighbors
# Parameter setting
k <- 30

# Prediction of test data
predKNN <-  knn(train=dxHm[indexTrain,], test=dxHm[indexTest,], 
                cl = dxHm[indexTrain, testGOs], k=k)

# Performance evaluation
confusionMatrix(predKNN, newClassGOHm[indexTest, drop=T])

# Parameter setting
ctrl <- trainControl(method = "cv", number = 5, savePredictions = "final")

modelKNN <- train(`GO.0042981`~ ., data = dxHm[indexTrain,], method = "knn", 
                  trControl = ctrl, tuneGrid = expand.grid(k = seq(1, 100, by = 10)), 
                  metric = "Accuracy")

# See results
print(modelKNN)

# Prediction of test data
predKNN<- predict(modelKNN, dxHm[indexTest,1:471])

# Performance evaluation
confusionMatrix(predKNN, newClassGOHm[indexTest, drop=T])

## Random Forest
# Parameter setting
mtry<-1

# Model generation
modelRF <-  randomForest(`GO.0042981` ~ ., data=dxHm[indexTrain,], importance=TRUE,
             proximity=TRUE, mtry=mtry, ntree=500)

# Prediction of test data
predRF <-predict(modelRF, dxHm[indexTest,1:471])

# Performance evaluation
confusionMatrix(predRF, newClassGOHm[indexTest, drop=T])

# Parameter setting
mtry<-75

# Model generation
modelRF <-  randomForest(`GO.0042981`~., data=dxHm[indexTrain,], mtry=mtry, ntree=500)

# Prediction of test data
predRF <-predict(modelRF, dxHm[indexTest,1:471])

# Performance evaluation
confusionMatrix(predRF, newClassGOHm[indexTest, drop=T])
    
## Support Vector Machine
# Model generation
modelSVM <- svm(`GO.0042981`~., data=dxHm[indexTrain,], kernel = "linear", 
                cost = 1, scale = FALSE)

# Model summary 
summary(modelSVM)

# Prediction of test data
predSVM <- predict(modelSVM, dxHm[indexTest,1:471], type = 'class')

# Performance evaluation
confusionMatrix(predSVM, newClassGOHm[indexTest, drop=T])

# Model generation
modelSVM <- svm(`GO.0042981` ~ ., data=dxHm[indexTrain,],
                kernel = "linear", cost = 100)

# Prediction of test data
predSVM <- predict(modelSVM, dxHm[indexTest,1:471], type = 'class')

# Performance evaluation
confusionMatrix(predSVM, newClassGOHm[indexTest, drop=T])

# Parameter setting
obj <- tune(svm, `GO.0042981`~., data = dxHm[indexTrain,], kernel="radial",
            ranges = list(gamma = seq(0.05,0.1,0.01), cost = seq(1,100,10)),
            tunecontrol = tune.control(sampling = "fix"))

# Model generation
modelSVM <- obj$best.model

# Prediction of test data
predSVM <- predict(modelSVM, dxHm[indexTest,1:(dim(dxHm)[2]-1)], type = 'class')

# Performance evaluation
confusionMatrix(predSVM, newClassGOHm[indexTest, drop=T])

# Multiclass classification

## Builing of the test and train datasets
# Choice of biological functions to predict
testGOs <- c("GO.0045087", "GO.0006914", "GO.0090398", "GO.0042981", "GO.0034599")

# Building of the GO target matrix
miniTableHmGO <- dxFeatures_GOs_Hm[, testGOs]
newClassGOHm <- as.factor(apply(miniTableHmGO, MARGIN = 1, FUN = function(x) sum(x * 2^(rev(seq(along = x)) - 1))))
dxHm <- dxFeatures_GOs_Hm[, !(names(dxFeatures_GOs_Hm) %in% testGOs)]

# Define the mapping from numeric values to GO-terms, only because they are independent
mapeoGO <- c(
    "1" = "GO.0034599",
    "2" = "GO.0042981",
    "4" = "GO.0090398",
    "8" = "GO.0006914",
    "16" = "GO.0045087"
)

newClassGOHm <- factor(newClassGOHm, levels = names(mapeoGO), labels = mapeoGO)
dxHm <- cbind(dxHm, newClassGOHm)
colnames(dxHm)[ncol(dxHm)] <- "classGO"

# Generation of test and train indices
indexData <- createFolds(newClassGOHm, k = 5)
indexTest <- indexData[[3]]
indexTrain <- setdiff(1:nrow(dxHm), indexTest)
   
## K-Nearest Neighbors
# Parameter setting
k <- 30

# Prediction of test data
predKNN <- knn(
    train = dxHm[indexTrain, -which(colnames(dxHm) == "classGO")], 
    test = dxHm[indexTest, -which(colnames(dxHm) == "classGO")], 
    cl = dxHm[indexTrain, "classGO"], k = k)

# Performance evaluation
confusionMatrix(predKNN, newClassGOHm[indexTest, drop=T])

## Random Forest
# Parameter setting
mtry<-1

# Model generation
modelRF <-  randomForest(classGO~., data=dxHm[indexTrain,], importance=TRUE,
                         mtry=mtry, ntree=500)

# Prediction of test data
predRF <-predict(modelRF, dxHm[indexTest,1:(ncol(dxHm)-1)])

# Performance evaluation
confusionMatrix(predRF, newClassGOHm[indexTest])

## Support Vector Machine

# Model generation
modelSVM <- svm(dxHm[indexTrain,1:(ncol(dxHm)-1)], dxHm[indexTrain, ncol(dxHm)], 
                kernel = "linear", cost = 1, scale = FALSE)

summary(modelSVM)

# Prediction of test data
predSVM <- predict(modelSVM, dxHm[indexTest,1:(ncol(dxHm)-1)], type = 'class')

# Performance evaluation
confusionMatrix(predSVM, newClassGOHm[indexTest])

# Model generation
modelSVM <- svm(dxHm[indexTrain,1:(ncol(dxHm)-1)], dxHm[indexTrain, ncol(dxHm)], 
                kernel = "radial", cost = 50, gamma= 0.01)

# Prediction of test data
predSVM <- predict(modelSVM, dxHm[indexTest,1:(ncol(dxHm)-1)], type = 'class')

# Performance evaluation
confusionMatrix(predSVM, newClassGOHm[indexTest])