# Databricks notebook source exported at Tue, 11 Oct 2016 01:55:12 UTC
# MAGIC %md
# MAGIC 
# MAGIC # Artificial Intelligence
# MAGIC 
# MAGIC ## Machine Learning with R. A basic introduction 
# MAGIC 
# MAGIC ![img](http://www.iqworkforce.com/wp-content/uploads/2015/11/AI.jpg)
# MAGIC 
# MAGIC ** Artificial Intelligence (AI) ** is intelligence exhibited by machines (https://en.wikipedia.org/wiki/Artificial_intelligence)
# MAGIC 
# MAGIC **Machine Learning** is an area of **Artificial Intelligence** concerned with the development of techniques that allow computers to learn. Learning is the ability of the machine to improve its performance based on previous results. (http://www.igi-global.com/dictionary/machine-learning/17656)

# COMMAND ----------

# First of all. We install the e1071 package, dependency for other packages
install.packages("e1071")
library(e1071)

# COMMAND ----------

# Get the list of installed packages
installed.packages()

# COMMAND ----------

# Get the list of available datasets
data()

# COMMAND ----------

# Create an R function to combine several ggplot graphs. Get from http://www.peterhaschke.com/r/2013/04/24/MultiPlot.html 
multiplot <- function(..., plotlist = NULL, file, cols = 1, layout = NULL) {
  require(grid)

  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  if (is.null(layout)) {
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

  if (numPlots == 1) {
    print(plots[[1]])

  } else {
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    for (i in 1:numPlots) {
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Type of data
# MAGIC 
# MAGIC * Numeric (num)
# MAGIC * Categorical (Factors)

# COMMAND ----------

str(iris)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Machine Learning
# MAGIC * ### Unsupervised
# MAGIC   + ** Clustering **
# MAGIC   + Association Rules
# MAGIC   + Sequential Patterns
# MAGIC   + PCA Principal Components Analysis
# MAGIC * ### Supervised
# MAGIC   + ** Prediction: Linear Regression **
# MAGIC   + Prediction: Non Linear Regression
# MAGIC   + Prediction: Regression with Trees
# MAGIC   + ** Classification: Logistic Regression (binary) **
# MAGIC   + ** Classification: Trees (multiclass) **

# COMMAND ----------

help(layout)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Unsupervised Methods: Clustering with k-means ##

# COMMAND ----------

library(ggplot2)
print(summary(iris))
print(str(iris))
par(mfrow=c(5,10))
qplot(data=iris, y=iris$Sepal.Length,x=iris$Species, geom="boxplot", main="Boxplot for type of Iris",xlab="Type of Iris", ylab="Sepal Length", colour=iris$Species) +
theme(plot.margin = unit(c(3, 3, 3, 3), "cm"))

# COMMAND ----------

# Clustering: K-Means
library(dplyr)


iris.mod <- iris %>% select(-Species)

set.seed(1234)
kmeans.clust <- kmeans(iris.mod, 3)
print(kmeans.clust)

layout(matrix(1), widths = lcm(15), heights = lcm(15))
plot(iris.mod %>% select(Petal.Length, Petal.Width), col = kmeans.clust$cluster)
points(as.data.frame(kmeans.clust$centers) %>% select(Petal.Length, Petal.Width),
col = 1:3, pch = 8, cex = 2)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Supervised Methods: Linear Regression with lm ##
# MAGIC ### Data inspection ###

# COMMAND ----------

# Wath the mtcars data structure
print(summary(mtcars))
str(mtcars)

# COMMAND ----------

# Information about the mtcars dataset
help(mtcars)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Exploration ###

# COMMAND ----------

# Show graph mpg vs wt
require(ggplot2)
# ggplot2 http://docs.ggplot2.org/current/ 

ggplot(data = mtcars, aes(x = wt, y = mpg)) + 
geom_point() + 
theme(plot.margin = unit(c(5, 5, 5, 5), "cm"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model construction ###

# COMMAND ----------

# Implementation with lm (alone). Build Model

print(cor(mtcars$mpg, mtcars$wt))
set.seed(13579)
training.index <- base::sample(x = 1:nrow(mtcars), size = 24, replace = FALSE)
mtcars.training <- mtcars[training.index,]
mtcars.test <- mtcars[-training.index,]

lm.model <- lm(mpg ~ wt ,data = mtcars.training)
summary(lm.model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction and Evaluation ###

# COMMAND ----------

# Implementation with lm alone. Predicting & Evaluation
# Predict using the test dataset
lm.model.predict <- predict(lm.model, mtcars.test)
print(head(lm.model.predict))

# Calculate rse with the formula
rse.train = sqrt(sum((lm.model$fitted - mtcars.training$mpg)^2) / (dim(mtcars.training)[1] - 2))
print(rse.train)

rse.test = sqrt(sum((lm.model.predict - mtcars.test$mpg)^2) / (dim(mtcars.test)[1] - 2))
print(rse.test)


# COMMAND ----------

# Represent the graph of the regression
library(ggplot2)
library(grid)

# Extract the intercept and slope with the coef() function
coef.intercept <- coef(lm.model)[1]
coef.slope <- coef(lm.model)[2]

g.training <- ggplot(data = mtcars.training, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_abline(slope = coef.slope, intercept = coef.intercept, color = "red") +
  ggtitle("Training") +
  coord_fixed(0.15)

g.test <- ggplot(data = mtcars.test, aes(x = wt, y = mpg)) +
  geom_point() +
  geom_abline(slope = coef.slope, intercept = coef.intercept, color = "green") +
  ggtitle("Test") +
  coord_fixed(0.18)

# par(mfrow=c(2,1))
multiplot(g.training, g.test, cols = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Implementation with caret ###

# COMMAND ----------

# Caret implementation + lm: building the model
library(caret)

set.seed(13579)
index.mtcars <- createDataPartition(mtcars$mpg, p=0.7, list=F)

train.mtcars <- mtcars[index.mtcars,]
test.mtcars <- mtcars[ -index.mtcars, ]

fitControl <- trainControl(method = "none")

lm.model.caret <- train(mpg ~ wt ,data = train.mtcars ,method = "lm", trControl = fitControl)
print(lm.model.caret)
print("Final Model")
summary(lm.model.caret$finalModel)

# COMMAND ----------

# Caret implementation + lm: Predicting & Evaluation
library(caret)
lm.model.caret.predict <- predict(lm.model.caret$finalModel, newdata = mtcars.test)
print(head(lm.model.caret.predict))

# Calculate rse with the formula
rse.train.caret = sqrt(sum((lm.model.caret$finalModel$fitted - train.mtcars$mpg)^2) / (dim(train.mtcars)[1] - 2))
print(paste("Training error: ", rse.train.caret))

rse.test.caret = sqrt(sum((lm.model.caret.predict - test.mtcars$mpg)^2) / (dim(test.mtcars)[1] - 2))
print(paste("Test error: ", rse.test.caret))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Supervised Methods: Logistic Regression (Binary Classification) with glmnet ##
# MAGIC ### Data Load & Inspection ###

# COMMAND ----------

# Get data and information about data from http://www.ats.ucla.edu/stat/r/dae/logit.htm 
admissions <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
str(admissions)

admissions$rank <- factor(admissions$rank)
str(admissions)

set.seed(21349)
training.adm.index <- base::sample(x = 1:nrow(admissions), size = 280, replace = FALSE)
admissions.train <- admissions[training.adm.index,]
admissions.test <- admissions[-training.adm.index,]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Building ###

# COMMAND ----------

# Building model with glmnet
library(glmnet)
library(dplyr)

# Use dplyr::select() to get the predictors
x.admissions <- admissions.train %>% select(-admit)
y.admissions <- admissions.train %>% select(admit)

mylogit <- glmnet(x = as.matrix(x.admissions), y = as.matrix(y.admissions), family = "binomial")

coef(mylogit, s=0.01)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Model building with caret ###

# COMMAND ----------

# Logistic regression with caret + glmnet
library(plyr)
library(dplyr)
library(caret)
library(glmnet)

admissions <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
str(admissions)

fitControl <- trainControl(classProbs = TRUE,
                           method = "cv",
                           number = 10,
                           summaryFunction = twoClassSummary
                          )
set.seed(21349)
index.admissions <- createDataPartition(admissions$admit, p=0.7, list=F)

train.admissions <- admissions[index.admissions,]
test.admissions <- admissions[ -index.admissions, ]

train.admissions$admit <- factor(train.admissions$admit)
test.admissions$admit <- factor(test.admissions$admit)
x.admissions <- train.admissions %>% select(-admit)
y.admissions <- train.admissions$admit

y.admissions <- revalue(y.admissions, c("1"="yes", "0"="no"))

# levels(y.admissions)

glmnet.caret.model <- train(x = x.admissions, y = y.admissions, method = "glmnet", metric="ROC", family="binomial", trControl = fitControl)
print(glmnet.caret.model)

coef(glmnet.caret.model$finalModel, s=0.1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction and Evaluation ###
# MAGIC 
# MAGIC ![img](http://lh3.ggpht.com/_qIDcOEX659I/SzjW6wGbmyI/AAAAAAAAAtY/Nls9tSN6DgU/contingency_thumb%5B3%5D.png?imgmax=800)

# COMMAND ----------

# Condition the test dataset
x.test.admissions <- test.admissions %>% select(-admit)
y.test.admissions <- revalue(test.admissions$admit, c("1"="yes", "0"="no"))
# Apply prediction
glmnet.caret.model.predict <- predict(glmnet.caret.model, newdata = x.test.admissions)
print(head(glmnet.caret.model.predict))

confusionMatrix(glmnet.caret.model.predict, y.test.admissions)


# COMMAND ----------

# Evaluation using the ROC curve
library(pROC)
roc(as.numeric(glmnet.caret.model.predict), as.numeric(y.test.admissions), percent=TRUE, plot=TRUE)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Supervised Methods. Binary Classification with gbm ##
# MAGIC ### Building the model ###

# COMMAND ----------

library(caret)
library(gbm)

train.admissions$rank <- factor(train.admissions$rank)
train.admissions$admit <- factor(train.admissions$admit)
test.admissions$rank <- factor(test.admissions$rank)
test.admissions$admit <- factor(test.admissions$admit)

print(head(train.admissions))

fitControl <- trainControl(classProbs = TRUE,
                           method = "repeatedcv",
                           number = 10,
                           repeats = 2,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

#fitControl <- trainControl(classProbs = TRUE,
#                           method = "cv",
#                           number = 10,
#                           summaryFunction = twoClassSummary
#                          )

#fitControl2 <- trainControl(method = "repeatedcv",
#                           number = 10,
#                           repeats = 2,
#                           classProbs = TRUE,
#                           summaryFunction = twoClassSummary,
#                           search = "random")

gbmGrid <-  expand.grid(shrinkage = c(0.1, 0.3, 0.5), n.minobsinnode = 10, interaction.depth = 3, n.trees = c(100, 150, 200))

train.admissions$admit <- revalue(train.admissions$admit, c("1"="yes", "0"="no"))

gbm.caret.model <- train(admit ~ ., data=train.admissions, method = "gbm", metric="ROC", trControl = fitControl, tuneGrid = gbmGrid, verbose=FALSE)

# gbmModel <- train(admit ~ ., data=admissions, method = "gbm", metric="ROC", trControl = fitControl, tuneLength = 10, verbose=FALSE)

gbm.caret.model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction and Evaluation ###

# COMMAND ----------

# Condition the test dataset
x.test.admissions <- test.admissions %>% select(-admit)
y.test.admissions <- revalue(test.admissions$admit, c("1"="yes", "0"="no"))
# Apply prediction
gbm.caret.model.predict <- predict(gbm.caret.model, newdata = x.test.admissions)
print(head(gbm.caret.model.predict))

confusionMatrix(gbm.caret.model.predict, y.test.admissions)

# COMMAND ----------

# Evaluation using the ROC curve
library(pROC)

roc(as.numeric(gbm.caret.model.predict), as.numeric(y.test.addmissions), percent=TRUE, plot=FALSE)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Comparison of Models with Caret ##

# COMMAND ----------

# Compare the glmnet & gbm models

models <- list( glmnet.caret.model, gbm.caret.model )
compar.models <- resamples( models )
summary( compar.models )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Supervised Methods. Multiclass Classification with gbm ##

# COMMAND ----------

library(caret)
library(gbm)
library(e1071)

fitControl <- trainControl(classProbs = TRUE,
                           method="repeatedcv",
                           number=5,
                           repeats=1,
                           summaryFunction = defaultSummary,
                           verboseIter=TRUE)

set.seed(825)
iris.index.train <- createDataPartition(iris$Species, p=0.7, list=F)
iris.train <- iris[iris.index.train,]
iris.test <- iris[-iris.index.train,]

gbm.caret.model <- train(Species ~ ., data=iris.train, method="gbm", verbose=FALSE)
gbm.caret.pred <- predict(gbm.caret.model, iris.test, n.trees=200, type="raw")

head(gbm.caret.pred)

confusionMatrix(gbm.caret.pred, iris.test$Species)



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Classification multiclass with trees: rpart ##

# COMMAND ----------

# Classification with caret & rpart
library(rpart)
library(caret)

rpart.model <- train(Species ~ ., data=iris, method="rpart", cp=0.1)

print(rpart.model)

plot(rpart.model$finalModel, uniform=TRUE, main="Classification Tree for Iris")
text(rpart.model$finalModel, use.n=TRUE, all=TRUE, cex=.8)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Classification multiclass with randomForest ##

# COMMAND ----------

library(randomForest)
library(caret)
# Set param to fix value
#control <- trainControl(method="repeatedcv", number=10, repeats=3)
#tunegrid <- expand.grid(mtry=3)

# Get value random
# control <- trainControl(method="repeatedcv", number=10, repeats=3, classProbs = TRUE, summaryFunction = twoClassSummary, search="random")

# Move over grid
control <- trainControl(method="cv", number=10)
tunegrid <- expand.grid(mtry=c(1:15))

rf.model <- train(Species ~ ., data=iris, method="rf", trControl = control, tuneGrid=tunegrid)

rf.model



# COMMAND ----------



# COMMAND ----------


