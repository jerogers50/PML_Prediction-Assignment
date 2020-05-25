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
library(caret) #version 6.0-86
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart) #version 4.1-15
library(corrplot) #version 0.84
```

```
## corrplot 0.84 loaded
```

```r
library(rattle) #version 5.3.0
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.3.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
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

# Create a partition using the training dataset
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
NA_variable    <- sapply(train, function(x) mean(is.na(x))) > 0.90
train <- train[, NA_variable==FALSE]
test  <- test[, NA_variable==FALSE]
dim(train)
dim(test)

head(train)
head(test)

# Remove identification / non-numerical variables. In this case it's the first 5 variables.
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
corrplot(cor_matrix, method = "square", tl.cex = .6, tl.col = rgb(0, 0, 0))
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
##          A 1511  251   44  119   74
##          B   42  626   33   32   68
##          C   17   92  840   94   76
##          D   96  118   68  669  154
##          E    8   52   41   50  710
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7402          
##                  95% CI : (0.7288, 0.7514)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6695          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9026   0.5496   0.8187   0.6940   0.6562
## Specificity            0.8841   0.9631   0.9426   0.9114   0.9686
## Pos Pred Value         0.7559   0.7815   0.7507   0.6054   0.8246
## Neg Pred Value         0.9581   0.8991   0.9610   0.9383   0.9260
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2568   0.1064   0.1427   0.1137   0.1206
## Detection Prevalence   0.3397   0.1361   0.1901   0.1878   0.1463
## Balanced Accuracy      0.8934   0.7564   0.8806   0.8027   0.8124
```


```r
set.seed(1800)
# Create Random Forest model
control_random_forest <- trainControl(method="cv", number=5, verboseIter=FALSE)
model_fit_rf <- train(classe ~ ., data=train, method="rf", trControl=control_random_forest)
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
##         OOB estimate of  error rate: 0.25%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    1    0    0    0 0.0002560164
## B    8 2646    3    1    0 0.0045146727
## C    0    7 2389    0    0 0.0029215359
## D    0    1    7 2244    0 0.0035523979
## E    0    1    0    6 2518 0.0027722772
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
##          A 1674    0    0    0    0
##          B    0 1137    9    0    0
##          C    0    2 1017    7    0
##          D    0    0    0  957    6
##          E    0    0    0    0 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9959          
##                  95% CI : (0.9939, 0.9974)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9948          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9982   0.9912   0.9927   0.9945
## Specificity            1.0000   0.9981   0.9981   0.9988   1.0000
## Pos Pred Value         1.0000   0.9921   0.9912   0.9938   1.0000
## Neg Pred Value         1.0000   0.9996   0.9981   0.9986   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1932   0.1728   0.1626   0.1828
## Detection Prevalence   0.2845   0.1947   0.1743   0.1636   0.1828
## Balanced Accuracy      1.0000   0.9982   0.9947   0.9958   0.9972
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
