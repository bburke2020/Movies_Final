library(lmridge) #FOR lmridge()
library(broom) #FOR glance() AND tidy()
library(MASS) #FOR lm.ridge()
library(ggplot2) #FOR ggplot()
library(tidymodels)
library(baguette) #FOR BAGGED TREES
library(xgboost) #FOR GRADIENT BOOSTING
library(caret) #FOR confusionMatrix()
library(vip) 

#clean data#
movies_1 <- na.omit(movies)
set.seed(123, kind = NULL, normal.kind = NULL, sample.kind = NULL)
#Partition Data#
seed(123)
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
Model3 <- lmridge(gross ~ I(budget) + I(budget^2), data = train, K = .01)
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


#Multivariate Regression#
Model5 <- lm(gross ~ budget+rating+genre+score+votes+runtime, data = train)
summary(Model5)

PRED_5_IN <- predict(Model5, train) 
PRED_5_OUT <- predict(Model5, test) 

(RMSE_5_IN<-sqrt(sum((PRED_5_IN-train$gross)^2)/length(PRED_5_IN))) #computes in-sample error
(RMSE_5_OUT<-sqrt(sum((PRED_5_OUT-test$gross)^2)/length(PRED_5_OUT)))

#Regularization#
Model6 <- lmridge(gross ~ budget+rating+genre+score+votes+runtime, data = train, k = .01)
summary(Model6)

PRED_6_IN <- predict(Model6, train) 
PRED_6_OUT <- predict(Model6, test) 

(RMSE_6_IN<-sqrt(sum((PRED_6_IN-train$gross)^2)/length(PRED_6_IN))) #computes in-sample error
(RMSE_6_OUT<-sqrt(sum((PRED_6_OUT-test$gross)^2)/length(PRED_6_OUT)))





