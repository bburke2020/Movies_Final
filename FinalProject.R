library(lmridge) #FOR lmridge()
library(broom) #FOR glance() AND tidy()
library(MASS) #FOR lm.ridge()
library(ggplot2) #FOR ggplot()
library(tidymodels)
library(baguette) #FOR BAGGED TREES
library(xgboost) #FOR GRADIENT BOOSTING
library(caret) #FOR confusionMatrix()
library(vip) 
library(mgcv)
#clean data#
movies_1 <- na.omit(movies)
set.seed(100)
#Partition Data#
set.seed(120)
movies_1$budget_2 <- movies_1$budget^2
split<-initial_split(movies_1, prop=.7)
train<-training(split)
holdout<-testing(split)

split_2 <- initial_split(holdout, prop = .5)
test <- training(split_2)
validation <- testing(split_2)

#Data Correlation#                                      
cor(movies_1[,c(4,6,7,12,13,15)])                    

#Strongest Relathionship(Gross and Budget)#
Model1 <- lm(gross ~ budget, data = train)
summary(Model1)

ggplot(movies_1, aes(x = budget, y = gross)) +
  geom_point() +
  geom_smooth(method = 'lm')

PRED_1_IN <- predict(Model1, train) 
PRED_1_OUT <- predict(Model1, test) 

(RMSE_1_IN<-sqrt(sum((PRED_1_IN-train$gross)^2)/length(PRED_1_IN))) #computes in-sample error
(RMSE_1_OUT<-sqrt(sum((PRED_1_OUT-test$gross)^2)/length(PRED_1_OUT)))

hist(Model1$residuals)
jarque.bera.test(Model1$residuals)


#Model 2 with nonlinear transformation#

Model2 <- lm(gross ~ I(budget) + I(budget_2), data = train)
summary(Model2)
ggplot(movies_1, aes(x = budget^2, y = gross)) +
  geom_point() +
  geom_smooth(method = 'lm')

PRED_2_IN <- predict(Model2, train) 
PRED_2_OUT <- predict(Model2, test) 

(RMSE_2_IN<-sqrt(sum((PRED_2_IN-train$gross)^2)/length(PRED_2_IN))) #computes in-sample error
(RMSE_2_OUT<-sqrt(sum((PRED_2_OUT-test$gross)^2)/length(PRED_2_OUT)))

hist(Model1$residuals)
jarque.bera.test(Model1$residuals)


TABLE_VAL <- as.table(matrix(c(RMSE_1_IN,RMSE_2_IN, RMSE_1_OUT,RMSE_2_OUT), ncol=2, byrow=TRUE))
colnames(TABLE_VAL) <- c('Model 1','Model 2')
rownames(TABLE_VAL) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS

#Regularization#
Model3 <- lmridge(gross ~ I(budget) + I(budget^2), data = train, K = seq(.01,.05,.1))
summary(Model3)

PRED_3_IN <- predict(Model3, train) 
PRED_3_OUT <- predict(Model3, test) 

(RMSE_3_IN<-sqrt(sum((PRED_3_IN-train$gross)^2)/length(PRED_3_IN))) #computes in-sample error
(RMSE_3_OUT<-sqrt(sum((PRED_3_OUT-test$gross)^2)/length(PRED_3_OUT)))

TABLE_VAL <- as.table(matrix(c(RMSE_1_IN,RMSE_2_IN, RMSE_3_IN, RMSE_1_OUT,RMSE_2_OUT, RMSE_3_OUT), ncol=3, byrow=TRUE))
colnames(TABLE_VAL) <- c('Model 1','Model 2', 'Model 3')
rownames(TABLE_VAL) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS

#GAM Moedl#
Model4 <- gam(gross ~ s(budget), data = train, family = "gaussian")
summary(Model4)

PRED_4_IN <- predict(Model4, train) 
PRED_4_OUT <- predict(Model4, test) 

(RMSE_4_IN<-sqrt(sum((PRED_4_IN-train$gross)^2)/length(PRED_4_IN))) #computes in-sample error
(RMSE_4_OUT<-sqrt(sum((PRED_4_OUT-test$gross)^2)/length(PRED_4_OUT)))

TABLE_VAL <- as.table(matrix(c(RMSE_1_IN,RMSE_2_IN, RMSE_3_IN, RMSE_4_IN, RMSE_1_OUT,RMSE_2_OUT, RMSE_3_OUT, RMSE_4_OUT), ncol=4, byrow=TRUE))
colnames(TABLE_VAL) <- c('Model 1','Model 2', 'Model 3', 'Model 4')
rownames(TABLE_VAL) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS

#Plot the 4 models#

x_grid <- seq(0,15000000000,1000) #CREATES GRID OF X-AXIS VALUES
plot(train$gross ~ train$budget, col='blue')
predictions_1 <- predict(Model1, list(budget = x_grid)) 
predictions_2 <- predict(Model2, list(budget = x_grid, budget_2 = x_grid^2))
predictions_3 <- predict(Model3, list(budget = x_grid))
predictions_4 <- predict(Model4, list(budget = x_grid))
lines(x_grid, predictions_1, col='lightblue', lwd=3) 
lines(x_grid, predictions_2, col='purple', lwd=3) 
lines(x_grid, predictions_3, col='green', lwd=3) 
lines(x_grid, predictions_4, col='darkgreen', lwd=3) 
points(test$gross ~ test$budget, col='red', pch=3, cex=.5)
points(validation$gross ~ validation$budget, col = 'orange', pch = 2, cex = .5)

#Tabel#
TABLE_VAL <- as.table(matrix(c(RMSE_1_IN,RMSE_2_IN, RMSE_3_IN, RMSE_4_IN, RMSE_1_OUT,RMSE_2_OUT, RMSE_3_OUT, RMSE_4_OUT), ncol=4, byrow=TRUE))
colnames(TABLE_VAL) <- c('Model 1','Model 2', 'Model 3', 'Model 4')
rownames(TABLE_VAL) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL


PRED_1_Val <- predict(Model1, validation)
PRED_2_Val <- predict(Model2, validation)
PRED_3_Val <- predict(Model3, validation)
PRED_4_Val <- predict(Model4, validation)
(RMSE_1_Val<-sqrt(sum((PRED_1_Val-validation$gross)^2)/length(PRED_1_Val)))
(RMSE_2_Val<-sqrt(sum((PRED_2_Val-validation$gross)^2)/length(PRED_2_Val)))
(RMSE_3_Val<-sqrt(sum((PRED_3_Val-validation$gross)^2)/length(PRED_3_Val)))
(RMSE_4_Val<-sqrt(sum((PRED_4_Val-validation$gross)^2)/length(PRED_4_Val)))

TABLE_VALIDATION <- as.table(matrix(c(RMSE_1_Val, RMSE_2_Val, RMSE_3_Val, RMSE_4_Val), ncol=4, byrow=TRUE))
colnames(TABLE_VALIDATION) <- c('Model 1','Model 2', 'Model 3', 'Model 4')
rownames(TABLE_VALIDATION) <- c('RMSE_Val')
TABLE_VALIDATION

#Model 2 is the best#

movies_1$rating <- as.factor(movies_1$rating)

#Multivariate Regression#
Model5 <- lm(gross ~ budget+rating+score+genre+votes+runtime, data = train)#rating
summary(Model5)

PRED_5_IN <- predict(Model5, train) 
PRED_5_OUT <- predict(Model5, test) 

(RMSE_5_IN<-sqrt(sum((PRED_5_IN-train$gross)^2)/length(PRED_5_IN))) #computes in-sample error
(RMSE_5_OUT<-sqrt(sum((PRED_5_OUT-test$gross)^2)/length(PRED_5_OUT)))

#Regularization#
Model6 <-lmridge(gross ~ I(budget) + I(budget^2) + I(budget^3) + I(budget^4) + I(budget^5), data = train, k = seq(.01,.05,.1))
summary(Model6)

View(Model6)
PRED_6_IN <- predict(Model6, train) 
PRED_6_OUT <- predict(Model6, test) 

(RMSE_6_IN<-sqrt(sum((PRED_6_IN-train$gross)^2)/length(PRED_6_IN))) #computes in-sample error
(RMSE_6_OUT<-sqrt(sum((PRED_6_OUT-test$gross)^2)/length(PRED_6_OUT)))

#non-linear transformations
Model7 <- lm(gross ~ poly(budget, degree = 3) + genre + score + votes + runtime, data = train) #rating
summary(Model7)

PRED_7_IN <- predict(Model7, train)
PRED_7_OUT <- predict(Model7, test)

(RMSE_7_IN <- sqrt(sum((PRED_7_IN - train$gross)^2) / length(PRED_7_IN)))  # in-sample error
(RMSE_7_OUT <- sqrt(sum((PRED_7_OUT - test$gross)^2) / length(PRED_7_OUT)))





#SVM
library(e1071)
library(rsample)
library(rlang)


kern_type<-"radial"

SVM_Model <- svm(gross ~ budget + score + votes + runtime,
                 data = train,
                 type = "eps-regression",  # set to "eps-regression" for numeric prediction
                 kernel = kern_type,
                 cost = 1,  # REGULARIZATION PARAMETER
                 gamma = 1 / (ncol(train) - 1),  # DEFAULT KERNEL PARAMETER
                 coef0 = 0,  # DEFAULT KERNEL PARAMETER
                 degree = 2,  # POLYNOMIAL KERNEL PARAMETER
                 scale = FALSE)  # RESCALE DATA? (SET TO TRUE TO NORMALIZE)

print(SVM_Model)

PRED_8_IN <- predict(SVM_Model, train)
PRED_8_OUT <- predict(SVM_Model, test)

(RMSE_8_IN <- sqrt(sum((PRED_8_IN - train$gross)^2) / length(PRED_8_IN)))  # in-sample error
(RMSE_8_OUT <- sqrt(sum((PRED_8_OUT - test$gross)^2) / length(PRED_8_OUT)))


TABLE_Multi <- as.table(matrix(c(RMSE_5_IN,RMSE_6_IN, RMSE_7_IN, RMSE_8_IN, RMSE_5_OUT,RMSE_6_OUT, RMSE_7_OUT, RMSE_8_OUT), ncol=4, byrow=TRUE))
colnames(TABLE_Multi) <- c('Model 5','Model 6', 'Model 7', 'Model 8')
rownames(TABLE_Multi) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_Multi
TABLE_VAL

#TUNING THE SVM BY CROSS-VALIDATION
str(train)

tune_control<-tune.control(cross=10) #SET K-FOLD CV PARAMETERS
TUNE <- tune.svm(x = train[,c(6,7,12,15)],
                 y = train[,13],
                 type = "eps-regression",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(.01, .1, 1, 5, 10, 25, 50, 75, 100, 1000), #REGULARIZATION PARAMETER
                 gamma = 1/(ncol(train)-1), #KERNEL PARAMETER
                 coef0 = 0,           #KERNEL PARAMETER
                 degree = 2)          #POLYNOMIAL KERNEL PARAMETER

print(TUNE)


  SVM_Model2 <- svm(gross ~ budget + score + votes + runtime,
                 data = train,
                 type = "eps-regression",  # set to "eps-regression" for numeric prediction
                 kernel = kern_type,
                 cost = TUNE$best.parameters$cost,  # REGULARIZATION PARAMETER
                 gamma = TUNE$best.parameters$gamma,  # DEFAULT KERNEL PARAMETER
                 coef0 = TUNE$best.parameters$coef0,  # DEFAULT KERNEL PARAMETER
                 degree = TUNE$best.parameters$degree,  # POLYNOMIAL KERNEL PARAMETER
                 scale = FALSE)  # RESCALE DATA? (SET TO TRUE TO NORMALIZE)

PRED_9_IN <- predict(SVM_Model2, train)
PRED_9_OUT <- predict(SVM_Model2, test)

(RMSE_9_IN <- sqrt(sum((PRED_9_IN - train$gross)^2) / length(PRED_9_IN)))  # in-sample error
(RMSE_9_OUT <- sqrt(sum((PRED_9_OUT - test$gross)^2) / length(PRED_9_OUT)))




#SPECIFYING THE CLASSIFICATION TREE MODEL
class_spec <- decision_tree(min_n = 20 , #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("regression")
print(class_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
class_fmla <- budget ~ rating + genre + score + votes + runtime
class_tree <- class_spec %>%
  fit(formula = class_fmla, data = train)
print(class_tree)

#VISUALIZING THE CLASSIFICATION TREE MODEL:
class_tree$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

plotcp(class_tree$fit)