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
reg_spec <- decision_tree(min_n = 20 , #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("regression")
print(class_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
rg_fmla <- gross ~ rating + genre + score + votes + runtime + budget
reg_tree <- class_spec %>%
  fit(formula = class_fmla, data = train)
print(reg_tree)

#VISUALIZING THE CLASSIFICATION TREE MODEL:
reg_tree$fit %>%
  rpart.plot(type = 4, roundint = FALSE)

plotcp(reg_tree$fit)


pred_10_in <- predict(reg_tree, new_data = train) %>%
  bind_cols(train)
pred_10_out <- predict(reg_tree, new_data = test) %>%
  bind_cols(test)

RMSE_10_Out <- rmse(pred_10_out, estimate=.pred, truth=gross)
RMSE_10_In <- rmse(pred_10_in, estimate=.pred, truth=gross)
RMSE_10_In
RMSE_10_Out

#Tuning Decision Tree#
Model <- gross ~ budget + score + votes + runtime
tree_spec <- decision_tree(min_n = tune(),
                           tree_depth = tune(),
                           cost_complexity= tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")
tree_grid <- grid_regular(parameters(tree_spec), levels = 3)
tune_results <- tune_grid(tree_spec,
                          Model, #MODEL FORMULA
                          resamples = vfold_cv(train, v=3), #RESAMPLES / FOLDS
                          grid = tree_grid, #GRID
                          metrics = metric_set(rmse)) #BENCHMARK METRIC
#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_params <- select_best(tune_results)

#FINALIZE THE MODEL SPECIFICATION
final_spec <- finalize_model(tree_spec, best_params)

#FIT THE FINALIZED MODEL
final_model <- final_spec %>% fit(Model, train)
pred_11_in <- predict(final_model, new_data = train) %>%
  bind_cols(train)
pred_11_out <- predict(final_model, new_data = test) %>%
  bind_cols(test) 

RMSE_11_Out <- rmse(pred_11_out, estimate=.pred, truth=gross)
RMSE_11_In <- rmse(pred_11_in, estimate=.pred, truth=gross)
RMSE_11_In
RMSE_11_Out


#Bagged Tree Model#
spec_bagged <- bag_tree(min_n = 20 , #minimum number of observations for split
                        tree_depth = 30, #max tree depth
                        cost_complexity = 0.01, #regularization parameter
                        class_cost = NULL)  %>% #for output class imbalance adjustment (binary data only)
  set_mode("regression") %>% #can set to regression for numeric prediction
  set_engine("rpart", times=100) #times = # OF ENSEMBLE MEMBERS IN FOREST
spec_bagged
bagged_forest <- spec_bagged %>%
  fit(formula = Model, data = train)
print(bagged_forest)
pred_12_in <- predict(bagged_forest, new_data = train) %>%
  bind_cols(train)
pred_12_out <- predict(bagged_forest, new_data = test) %>%
  bind_cols(test)

RMSE_12_Out <- rmse(pred_12_out, estimate=.pred, truth=gross)
RMSE_12_In <- rmse(pred_12_in, estimate=.pred, truth=gross)
RMSE_12_In
RMSE_12_Out

spec_bagged_2 <- bag_tree(min_n = tune() , #minimum number of observations for split
                          tree_depth = tune(), #max tree depth
                          cost_complexity = tune(), #regularization parameter
                          class_cost = tune())  %>% #for output class imbalance adjustment (binary data only)
  set_mode("regression") %>% #can set to regression for numeric prediction
  set_engine("rpart", times=100) #times = # OF ENSEMBLE MEMBERS IN FOREST
spec_bagged_2

tree_grid <- grid_regular(parameters(spec_bagged_2), levels = 3)
tune_results <- tune_grid(spec_bagged_2,
                          Model, #MODEL FORMULA
                          resamples = vfold_cv(train, v=3), #RESAMPLES / FOLDS
                          grid = tree_grid, #GRID
                          metrics = metric_set(rmse)) #BENCHMARK METRIC
#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_params <- select_best(tune_results)

#FINALIZE THE MODEL SPECIFICATION
final_spec <- finalize_model(tree_spec, best_params)



help("cbind")
library(dplyr)
oscars_df <- mutate(oscars_df, Winner = ifelse(oscars_df$Award == "Winner",1,0)) 
colnames(movies_1)[colnames(movies_1) == "name"] <- "Film"                        
merged_df <- left_join(oscars_df,movies_1, by = "Film")                          
merged_2 <- subset(merged_df, !is.na(budget))
merged_3 <- na.omit(merged_df)
Oscars<- merged_2








#Classification#
set.seed(120)
split<-initial_split(Oscars, prop=.7)
train2<-training(split)
holdout2<-testing(split)

split_2 <- initial_split(holdout2, prop = .5)
test2 <- training(split_2)
validation2 <- testing(split_2)

#Logit Model#
Model_13 <- glm(Winner ~ budget + gross + votes + score + runtime, data = train2, family = binomial(link="logit"))
summary(Model_13)

Test_Stat<-Model_13$null.deviance-Model_13$deviance #difference in deviance
Test_df<-Model_13$df.null-Model_13$df.residual #difference in degrees of freedom
1-pchisq(Test_Stat, Test_df) #p-value for null hypothesis H_0:
#the x-variables are not useful predictors of the categorical variable

Predictions_13_In <- predict(Model_13, train2, type = "response")
Predictions_13_Out <- predict(Model_13, test2, type = "response")

confusion<-table(Predictions_13_In, train2$Winner == 1)
confusion

confusion_out<-table(Predictions_13_Out, test2$Winner == 1)

confusionMatrix(confusion, positive='TRUE')
confusionMatrix(table(Predictions_13_In >= 0.5, train2$Winner == 1), positive='TRUE')

confusionMatrix(confusion_out, positive='TRUE')
confusionMatrix(table(Predictions_13_Out >= 0.5, test2$Winner == 1), positive='TRUE')


#Porbit Model#
Model_14 <- glm(Winner ~ budget + gross + votes + score + runtime, data = train2, family = binomial(link="probit"))
summary(Model_14)

Predictions_14_In <- predict(Model_14, train2, type = "response")
Predictions_14_Out <- predict(Model_14, test2, type = "response")

confusion3<-table(Predictions_14_In, train2$Winner == 1)
confusion

confusion4<-table(Predictions_13_Out, test2$Winner == 1)

confusionMatrix(confusion3, positive='TRUE')
confusionMatrix(table(Predictions_14_In >= 0.5, train2$Winner == 1), positive='TRUE')

confusionMatrix(confusion4, positive='TRUE')
confusionMatrix(table(Predictions_14_Out >= 0.5, test2$Winner == 1), positive='TRUE')

#ROC and AUC#
library(tidymodels)
library(pROC)
pva <- data.frame(preds = Predictions_13_In, acutal = factor(train2$Winner))
roc_obj <- roc(pva$acutal, pva$preds)
plot(roc_obj, col = "blue", main = "ROC Curve")

auc <- auc(roc_obj)
auc

pva2 <- data.frame(preds = Predictions_14_In, acutal = factor(train2$Winner))
roc_obj <- roc(pva2$acutal, pva2$preds)
plot(roc_obj, col = "red", main = "ROC Curve")

auc <- auc(roc_obj)
auc

#SVM Classfication#
kern_type <- "radial"

SVM_Model15<- svm(Winner ~ budget + gross + votes + score + runtime, 
                data = train2, 
                type = "C-classification", #set to "eps-regression" for numeric prediction
                kernel = kern_type,
                cost=1,                   #REGULARIZATION PARAMETER
                gamma = 1/(ncol(training)-1), #DEFAULT KERNEL PARAMETER
                coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                scale = FALSE)                #RESCALE DATA? (SET TO TRUE TO NORMALIZE)
summary(SVM_Model15)

(E_IN_15<-1-mean(predict(SVM_Model15, train2)==train2$Winner))
(E_OUT_15<-1-mean(predict(SVM_Model15, test2)==test2$Winner))

tunxe_control<-tune.control(cross=10) #
TUNE <- tune.svm(x = train2[,c(36,37,42,43,45)],
                 y = factor(train2$Winner),
                 type = "C-classification",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(.01, .1, 1, 10, 100, 1000), #REGULARIZATION PARAMETER
                 gamma = 1/(ncol(train2)-1), #KERNEL PARAMETER
                 coef0 = 0,           #KERNEL PARAMETER
                 degree = 2) 

print(TUNE)
library(tidymodels)
library(caret)
SVM_Retune15<- svm(Winner ~ budget + gross + votes + score + runtime, 
                   data = train2, 
                   type = "C-classification", 
                   kernel = kern_type,
                   degree = TUNE$best.parameters$degree,
                   gamma = TUNE$best.parameters$gamma,
                   coef0 = TUNE$best.parameters$coef0,
                   cost = TUNE$best.parameters$cost,
                   scale = FALSE)
(E_IN_RETUNE<-1-mean(predict(SVM_Retune15, train2)==train2$Winner))
(E_OUT_RETUNE<-1-mean(predict(SVM_Retune15, test2)==test2$Winner))


#Classfication Tree#
library(rpart.plot)
class_spec <- decision_tree(min_n = 20 , #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec)

class_16 <- factor(Winner) ~ gross + votes + runtime + score
class_tree16 <- class_spec %>%
  fit(formula = class_16, data = train2)
print(class_tree16)

class_tree16$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

plotcp(class_tree16$fit)

pred_class16 <- predict(class_tree16, new_data = train2, type="class") %>%
  bind_cols(train2) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

pred_prob16 <- predict(class_tree16, new_data = train2, type="prob") %>%
  bind_cols(test)
confusion <- table(pred_class16$.pred_class, pred_class16$Winner)
confusionMatrix(confusion, positive = "1")
