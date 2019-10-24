---
title: "Predicting the correct way to exercice"
author: "Anna-Lea Lesage"
date: "October 23, 2019"
output:
  html_document:
    keep_md: yes
---



## Summary

Is it possible to infer from the way a person is doing an exercice whether they are doing it right?
To answer this question, I used the existing weight lifting dataset. After selecting specific columns as predictors, I fitted a boosted tree model to the class variable. The in-sample error was very small, less than 5%, suggesting a possible over-fitting of the model. To verify this assumption, I checked whether some columns were correlated to others, and removed those which presented a high correlation rate ( greater than 80%). Still the new model performed very well and had an out-of sample error of about 6%. The last model was used to predict on hand of the given test set the class of the exercise.

## Data acquisition and cleaning

In order to properly determine whether an exercice is done correctly, 6 participants were asked to perform simple weight lifting exercices. They were asked to perform the exercice correctly on one set, and wrongly on four other sets. Each participant was fitted with 3 sensors (forearm, arm and belt), and a last sensor was located on the dumbbell. All the measurements have been compiled together into one table.

The data has been downloaded and saved in the working directory. It is already split into a training and a test set. We set the test set aside for the moment. It'll be used in the last section for the prediction.

The training data consists of about 20000 observations of 160 variables.


```r
data <- read.csv("pml-training.csv", as.is=TRUE, na.strings=c("", " ", NA))
dim(data)
```

```
## [1] 19622   160
```
The variable *classe* is our outcome. It defines whether the exercice has been performed correstly, class A, or incorrectly, classes B to E.  

### Data Preparation

We separate the data into a training and a test set. The provided test set will serve as validation set to verify the out-of-sample errors.

```r
library(caret)
inTrain <- createDataPartition(data$classe, p=0.7, list=FALSE)
training <- data[inTrain, ]
testing <- data[-inTrain, ]
```

Using the `str` function of R, we can have a quick look at the data. It seems that many columns do not have any entries of very few of them. We further investigate this using the following function:


```r
nonzero.fraction <- function(X) {
    nonzero.counts <- length(which(X != 0))
    nonzero.counts / length(X)
}
nonzero.columns <- lapply(training, nonzero.fraction)
non.zero.columns <- sapply(nonzero.columns, function(X) {X > 0.8}, simplify=TRUE)
```

I decided to keep all the columns which have more than 80% of non-zero values. The rest will not be used in the model building since the data is too sparse. This step reduces the number of potential variables to 59.


```r
training <- training[, non.zero.columns]
dim(training)
```

```
## [1] 13737    56
```

### Column selection

The number of predictors is still very high. Some of the columns could be discarded from the model:

* The user name, is not relevant for the prediction since we want to be able to predict the correctness of the mouvement for any user
* The raw_timestamp_part_1, is the duplicate of the cvtd_timestamp. We can safely discard one (or both)
* the new_window and num_window columns,
* the X column, as it represents the number of rows.


```r
discarded.columns <- c("X", "user_name", "raw_timestamp_part_1",
                       "raw_timestamp_part_2", "new_window", "num_window",
                       "cvtd_timestamp")
library(dplyr)
training <- training %>% select(-discarded.columns)
training$classe <- as.factor(training$classe)
```


## Model Selection

I normalize the data in the pre-processing step.
For the model I have chosen a boosted tree model as it allows both regression and classification.
Cross-validation is performed within the train function. I've set it to use a K-fold method using 10 folds (which means that each test fold has nearly a 1300 data points).

Furthermore, to improve the speed of the calculation, I'll allow parallel processing within the caret package. For this I use the packages *parallel* and *doParallel*.

```r
cluster <- parallel::makeCluster(parallel::detectCores() - 1)
doParallel::registerDoParallel()
```


```r
cross.validation.params <- trainControl(## 10-fold
    method="cv",
    number=10,
    allowParallel = TRUE  ## To use the multiprocessing cluster created ealier.
)
model <- train(classe ~ ., data=training, method="gbm", 
               preProcess=c("center", "scale"),
               trControl=cross.validation.params,
               verbose=FALSE
               )
parallel::stopCluster(cluster)
```

## Accuracy of the training set

With the cross-validation, we can already have a decent idea of the in-sample error. Using the *gbm* model gives also the possibility to visualize the accuracy of the model for different settings. Here I kept to the default settings for the interaction depth, number of tree and a shrinkage of 10%.


```r
plot(model)
```

![](assignment_files/figure-html/unnamed-chunk-8-1.png)<!-- -->
We see that we have a very good accuracy (over 95%) using only the training set. The best configuration is reached with 150 nodes in the tree and an interaction depth of 3.

```r
model.pred.train <- confusionMatrix.train(model)
model.pred.train
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 27.8  0.9  0.1  0.1  0.1
##          B  0.3 17.7  0.6  0.1  0.3
##          C  0.2  0.7 16.5  0.7  0.2
##          D  0.1  0.1  0.2 15.5  0.2
##          E  0.1  0.0  0.0  0.1 17.6
##                             
##  Accuracy (average) : 0.9513
```
### Verification of the out-sample error using the created test set.

The model performs really well on the training set. However, how does it perform on the test set we created in step 1.  


```r
pred.test <- predict(model, newdata=testing)
conf.pred.test <- confusionMatrix(as.factor(testing$class), pred.test)
conf.pred.test$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.564996e-01   9.449455e-01   9.509721e-01   9.615685e-01   2.888700e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   8.350398e-06
```
As expected, the accuracy is slightly less than on the training dataset but remains still above the 95%.

## Improvement of the model

Since we have already a pretty low in-sample error (less than 5%), we can ask ourself whether we are not overfitting the data. Have we selected the right predictors? Maybe some predictors are correlated to each other and will increase the bias of the model.
Let's look which columns are correlated and remove those from the predictors. 


```r
corrs <- cor(training[, -ncol(training)])  ## I remove the last column classe as it is a factor
highCorr <- findCorrelation(corrs, 0.8)  ## Find correlations above 0.8
## Those are the index of the correlated columns -> we'll remove those
training <- training[, -highCorr]
```

Let's calculate the model again and see if how it performs compared to the first one.

```r
cluster <- parallel::makeCluster(parallel::detectCores() - 1)
doParallel::registerDoParallel()
improved.model <- train(classe ~ ., data = training, method="gbm",
                        preProcess=c("center", "scale"),
                        trControl=cross.validation.params,
                        verbose=FALSE
                        )
parallel::stopCluster(cluster)
```



```r
impr.pred.test <- predict(improved.model, newdata=testing)
impr.conf.pred.test <- confusionMatrix(as.factor(testing$class), impr.pred.test)
impr.conf.pred.test$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.398471e-01   9.238325e-01   9.334661e-01   9.457876e-01   2.909091e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   1.201877e-12
```
The out-of sample error has increased a bit compared to the previous model made by using all the predictors regardless of their correlation. Indeed the accuracy of the first model was at over 95.6499575, 94.4945502, 95.0972069, 96.1568548, 28.8870008, 0, 8.3503982\times 10^{-4}%, now we have an accuracy of 93.9847069, 92.3832521, 93.3466058, 94.5787578, 29.0909091, 0, 1.2018773\times 10^{-10}%

## Prediction on the validation set

The validation set consists of 20 rows of data where we don't know the class the exercice. Let's see how our model predicts each case:


```r
validation <- read.csv("pml-testing.csv")
val.pred <- predict(improved.model, newdata=validation)
val.pred
```

```
##  [1] A A B A A C D B A A B C B A E E A B B B
## Levels: A B C D E
```


