---
title: "Prediction Assignment Writeup"
author: "Joel Rogers"
date: "5/20/2020"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---

# Assignment

You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

# Data Importing and Cleaning


```r
# Load packages
library(AppliedPredictiveModeling)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(gbm)
```

```
## Loaded gbm 2.1.5
```

```r
library(lubridate)
```

```
## Warning: package 'lubridate' was built under R version 3.6.2
```

```
## 
## Attaching package: 'lubridate'
```

```
## The following objects are masked from 'package:base':
## 
##     date, intersect, setdiff, union
```

```r
library(forecast)
```

```
## Warning: package 'forecast' was built under R version 3.6.2
```

```
## Registered S3 method overwritten by 'quantmod':
##   method            from
##   as.zoo.data.frame zoo
```

```r
library(e1071)
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(corrplot)
```

```
## corrplot 0.84 loaded
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.3.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```
## 
## Attaching package: 'rattle'
```

```
## The following object is masked from 'package:randomForest':
## 
##     importance
```

```r
library(modelr)
```

```
## Warning: package 'modelr' was built under R version 3.6.2
```

```r
# Download training data for this project
getwd()

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml_training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml_testing.csv")

# The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

# Read the activity data file 
training_data <- read.csv("pml_training.csv")
testing_data <- read.csv("pml_testing.csv")
head(training_data)
head(testing_data)

# Create a partition using the caret package with the training dataset on 70,30 ratio
train_variable <- createDataPartition(training_data$classe, p = 0.7, list = FALSE)
train <- training_data[train_variable, ]
test <- training_data[-train_variable, ]
dim(train)
dim(test)

# Remove variables with nearly zero variance
near_zero_variable <- nearZeroVar(train)
train <- train[, -near_zero_variable]
test  <- test[, -near_zero_variable]
dim(train)
dim(test)

# Remove variables with mostly NA observations
NA_variable    <- sapply(train, function(x) mean(is.na(x))) > 0.95
train <- train[, NA_variable==FALSE]
test  <- test[, NA_variable==FALSE]
dim(train)
dim(test)

head(train)
head(test)

# Remove identification / non-numerical variables
train <- train[, -(1:5)]
test  <- test[, -(1:5)]

head(train)
head(test)
```

# Correlation Analysis

Based on the plot below, it can be determined that there is little correlation across the variables. 


```r
ncol(train)
```

```
## [1] 54
```

```r
cor_matrix <- cor(train[, -ncol(train)])
corrplot(cor_matrix, method = "color", type = "lower", 
         tl.cex = .6, tl.col = rgb(0, 0, 0))
```

![](Prediction-Assignment_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

# Prediction Models

For the purpose of this analysis I developed Random Forest and Decision Tree predictive models. These models were used to determine which of the two would yield more accurate predictions on the testing dataset. 


```r
set.seed(1800)
# Create Decision Tree model
model_fit_dt <- rpart(classe ~ ., data=train, method="class")
fancyRpartPlot(model_fit_dt)
```

![](Prediction-Assignment_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

```r
# Apply Decision Tree prediction on test dataset to determine accuracy
predict_decision_tree <- predict(model_fit_dt, newdata=test, type="class")
confusion_matrix_dt <- confusionMatrix(predict_decision_tree, test$classe)
confusion_matrix_dt
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1519  179   30   73   22
##          B   34  635   36   15   17
##          C   31  116  847   71   12
##          D   79  174   98  741  137
##          E   11   35   15   64  894
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7878          
##                  95% CI : (0.7771, 0.7982)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7312          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9074   0.5575   0.8255   0.7687   0.8262
## Specificity            0.9278   0.9785   0.9527   0.9008   0.9740
## Pos Pred Value         0.8332   0.8616   0.7864   0.6029   0.8773
## Neg Pred Value         0.9618   0.9021   0.9628   0.9521   0.9614
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2581   0.1079   0.1439   0.1259   0.1519
## Detection Prevalence   0.3098   0.1252   0.1830   0.2088   0.1732
## Balanced Accuracy      0.9176   0.7680   0.8891   0.8348   0.9001
```


```r
set.seed(1800)
# Create Random Forest model
control_random_forest <- trainControl(method="cv", number=3, verboseIter=FALSE)
model_fit_rf <- train(classe ~ ., data=train, method="rf",
                          trControl=control_random_forest)
model_fit_rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.24%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    0    0    0    1 0.0002560164
## B    7 2648    3    0    0 0.0037622272
## C    0    4 2391    1    0 0.0020868114
## D    0    0   11 2241    0 0.0048845471
## E    0    1    1    4 2519 0.0023762376
```

```r
# Apply Randowm Forest prediction on test dataset to determine accuracy
predict_random_forest <- predict(model_fit_rf, newdata=test)
confusion_matrix_rf <- confusionMatrix(predict_random_forest, test$classe)
confusion_matrix_rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    2    0    0    0
##          B    1 1134    1    0    0
##          C    0    3 1025    4    0
##          D    0    0    0  959    2
##          E    0    0    0    1 1080
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9976         
##                  95% CI : (0.996, 0.9987)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.997          
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9956   0.9990   0.9948   0.9982
## Specificity            0.9995   0.9996   0.9986   0.9996   0.9998
## Pos Pred Value         0.9988   0.9982   0.9932   0.9979   0.9991
## Neg Pred Value         0.9998   0.9989   0.9998   0.9990   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1927   0.1742   0.1630   0.1835
## Detection Prevalence   0.2846   0.1930   0.1754   0.1633   0.1837
## Balanced Accuracy      0.9995   0.9976   0.9988   0.9972   0.9990
```

After comparing results from both models, the random forest model will be used for prediction on the testing dataset. With an accuracy of ~99% and an out of bounds error percentage or ~0.2% this is the best model to use. The accuracy of the decision tree was significantly lower.

# Predicting with the Testing Data


```r
# Apply random forest predictor to testing dataset
predict_final <- predict(model_fit_rf, newdata=testing_data)
predict_final
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
