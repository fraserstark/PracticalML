

## Summary 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These types of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of six participants. They were asked to perform barbell lifts correctly and incorrectly in five different ways. More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har).

The goal of this study is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. The training data for this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). The test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).


## Load required libraries

First of all we need to load the libraries we'll be using for this study. For model training, we'll make use of the `caret` package, which stands for *C*lassification *A*nd *Re*gression *T*raining.  



```r
library(caret)
library(corrplot)
library(e1071)
library(rpart)
library(rattle)
```


## Getting and loading the data


```r
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

train <- read.csv(url(train_url))
test <- read.csv(url(test_url))

dim(train); dim(test)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

We can see here that the training dataset contains 19622 observations - we'll use this to create a separate partition for validating our models later. We'll use the test data once we have evaluated the performance of our models. 

## Data preprocessing

As can be seen below, the first seven columns appear to be used to identify the participant in the study and are irrelevant for the purposes of our study. 


```r
names(train[,1:7])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

We can remove these from both sets of data. 


```r
train <- train[,-c(1:7)]
test <- test[,-c(1:7)]
```


Some of the variables in the dataset contain no variability, and these will not be useful when it comes to constructing prediction models. We can use the `nearZeroVar` function from caret to identify and remove these variables from our training and test datasets. 


```r
nzv <- nearZeroVar(train)

train <- train[-nzv]
test <- test[-nzv]
```

And we can also discard columns which contain mostly NA values, our threshold here is 95%


```r
na_cols <- sapply(train, function(x) mean(is.na(x))) > 0.95

train <- train[, na_cols==FALSE]
test <- test[,na_cols==FALSE]
```

The rest of the dataset includes some factor variables, but we want these to be numeric.


```r
numeric_cols <- colnames(train)[-ncol(train)]  # Select all but the last column
train[numeric_cols] <- sapply(train[numeric_cols], as.numeric)
```



```r
dim(train); dim(test)
```

```
## [1] 19622    53
```

```
## [1] 20 53
```


## Exploratory Data Analysis

Now that we've finished cleaning and preprocessing the data, we can use the `corrplot` package to visualise a correlation matrix. In the figure below, positive correlations between the variables are shown in larger dark blue circles, and negative correlations are shown in larger dark red circles.


```r
mat <- cor(train[,-53])
corrplot(mat, method = "circle", type = "lower", tl.cex = 0.7, tl.col = rgb(0,0,0))
```

![](MachineLearningAssignment_files/figure-html/unnamed-chunk-9-1.png)<!-- -->


## Predictive Modelling 

Let's take a step back and remind ourselves of the objective of the study. The goal is to use the data from the various sensors (visualised above), to make a prediction about the manner of the exercise - which is stored in our dataset as `classe`. There are many to choose from, but we will explore the use of three algorithms - Decision Trees, Random Forest, and Gradient Boosted Model.

Before we start training any models, we need to use the `createDataPartition` function (from the `caret` package) to set aside 30% of the training data into a new 'probe' dataset which we'll use to perform validation to assess how well the model might perform out of sample. 


```r
inTrain <- createDataPartition(train$classe, p = .7, list = FALSE)
train <- train[inTrain,]
probe <- train[-inTrain,]  # Validation dataset
```


### Decision Tree

Decision Trees are easy to interpret and generally perform better than regression models when the relationships are non linear. We've already loaded the caret packages and can fit a Decision Tree model like below. When we call `train` from the caret package, we tell it which model to fit (`method = "rpart"`), and the formula for the first argument (`classe ~ .`) which means we want to model `classe` as a function of all the other variables. 


```r
set.seed(521)
dtModel <- train(classe ~ ., 
                data = train, 
                method = "rpart", 
                trControl=trainControl(method = "none"), 
                tuneGrid=data.frame(cp=0.01)) 

dtModel
```

```
## CART 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: None
```



```r
fancyRpartPlot(dtModel$finalModel)
```

![](MachineLearningAssignment_files/figure-html/warning-FALSE-1.png)<!-- -->

We can use the `predict` function provided by caret to apply the model to our probe dataset, and then build a `confusionMatrix` to visualise the results. 


```r
dtPredict <- predict(dtModel, newdata = probe)

confusionMatrix(dtPredict, probe$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1082  134   18   48   31
##          B   46  456   95   76  143
##          C   23   94  561   94   72
##          D   41   64   51  410   43
##          E    4   11    0   43  475
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7252          
##                  95% CI : (0.7112, 0.7387)
##     No Information Rate : 0.2906          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6505          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9047   0.6008   0.7738  0.61103   0.6217
## Specificity            0.9209   0.8927   0.9165  0.94222   0.9827
## Pos Pred Value         0.8241   0.5588   0.6647  0.67323   0.8912
## Neg Pred Value         0.9593   0.9082   0.9499  0.92556   0.9193
## Prevalence             0.2906   0.1844   0.1762  0.16306   0.1857
## Detection Rate         0.2629   0.1108   0.1363  0.09964   0.1154
## Detection Prevalence   0.3191   0.1983   0.2051  0.14800   0.1295
## Balanced Accuracy      0.9128   0.7468   0.8452  0.77662   0.8022
```

We can see from these results that the Decision Tree model scores an accuracy of 0.75, which would result in an Out of Sample Error of 25%. 


### Random Forest

With Random Forest models we will use `trainControl` when fitting, which allows us to tune the parameters and we're going to use it to specify the resampling method (`method = "cv"`) for Cross Validation. This is then provided as a parameter to the call to `train`. 

The syntax and parameters for train are similar throughout, however this time we pass `rf` as the method. 


```r
tc <- trainControl(method = "cv", number = 3, verboseIter=FALSE)
```


```r
set.seed(521)
rfModel <- train(classe ~ .,
                 data = train,
                 method = "rf",
                 trControl = tc
                 )

rfModel
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 9158, 9157, 9159 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9882071  0.9850790
##   27    0.9883527  0.9852648
##   52    0.9807091  0.9755915
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

Let's look at the prediction results against the validation dataset.


```r
rfPredict <- predict(rfModel, newdata = probe)

confusionMatrix(rfPredict, probe$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1196    0    0    0    0
##          B    0  759    0    0    0
##          C    0    0  725    0    0
##          D    0    0    0  671    0
##          E    0    0    0    0  764
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2906     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2906   0.1844   0.1762   0.1631   0.1857
## Detection Rate         0.2906   0.1844   0.1762   0.1631   0.1857
## Detection Prevalence   0.2906   0.1844   0.1762   0.1631   0.1857
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

The Random Forest results yield a result of 100% accuracy. I would expect this to be high, but not quite perfect...so this should be investigated further and examined for overfitting. I'll run with this (for now) and make any corrections to the source code on GitHub, but often at times it's better to sacrifice a little accuracy for a more robust model which will perform well with new data.  


### Generalised Boosted Model

Better known as Gradient Boosting - boosting is another famous ensemble learning algorithm, where the aim is to reduce the high variance of learners by taking the average of lots of models fitted on bootstrapped data. 



```r
set.seed(521)
gbmModel <- train(classe ~ .,
                  data = train,
                  method = "gbm",
                  trControl = tc,
                  verbose = FALSE)

gbmModel
```

```
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 9158, 9157, 9159 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7506729  0.6838554
##   1                  100      0.8197565  0.7718342
##   1                  150      0.8533155  0.8143978
##   2                   50      0.8541157  0.8152045
##   2                  100      0.9066752  0.8818896
##   2                  150      0.9302611  0.9117395
##   3                   50      0.8958283  0.8680777
##   3                  100      0.9392150  0.9230677
##   3                  150      0.9592339  0.9484156
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```



```r
gbmPredict <- predict(gbmModel, newdata = probe)

confusionMatrix(gbmPredict, probe$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1186   15    0    0    1
##          B    6  733   16    4    4
##          C    2   11  701   23    7
##          D    2    0    8  642    9
##          E    0    0    0    2  743
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9733         
##                  95% CI : (0.9679, 0.978)
##     No Information Rate : 0.2906         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9661         
##                                          
##  Mcnemar's Test P-Value : 6.916e-05      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9916   0.9657   0.9669   0.9568   0.9725
## Specificity            0.9945   0.9911   0.9873   0.9945   0.9994
## Pos Pred Value         0.9867   0.9607   0.9422   0.9713   0.9973
## Neg Pred Value         0.9966   0.9922   0.9929   0.9916   0.9938
## Prevalence             0.2906   0.1844   0.1762   0.1631   0.1857
## Detection Rate         0.2882   0.1781   0.1704   0.1560   0.1806
## Detection Prevalence   0.2921   0.1854   0.1808   0.1606   0.1810
## Balanced Accuracy      0.9931   0.9784   0.9771   0.9756   0.9860
```

We can see from above that the final model used 150 trees at an interaction.depth of 3. Finally giving an accuracy of 97% for the Gradient Boosting model - again, this is expected to be high.


## Predicting against the Test dataset

Finally we can use our trained Random Forest model to make predictions against the test dataset provided with the study. 


```r
predict(rfModel, newdata = test)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The output above represents the predictions to the 20 observations in the test dataset, and is currently 100% accurate. 
