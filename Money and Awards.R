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

(E_IN_Model1<-1-mean(predict(Model1, train)==train$gross))

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
PRED_5_VAL <- predict(Model5, validation)

(RMSE_5_IN<-sqrt(sum((PRED_5_IN-train$gross)^2)/length(PRED_5_IN))) #computes in-sample error
(RMSE_5_OUT<-sqrt(sum((PRED_5_OUT-test$gross)^2)/length(PRED_5_OUT)))
(RMSE_5_VAL<-sqrt(sum((PRED_5_VAL-validation$gross)^2)/length(PRED_5_VAL)))
#Regularization#
Model6 <-lmridge(gross ~ I(budget) + I(budget^2) + I(budget^3) + I(budget^4) + I(budget^5), data = train, k = seq(.01,.05,.1))
summary(Model6)

View(Model6)
PRED_6_IN <- predict(Model6, train) 
PRED_6_OUT <- predict(Model6, test) 
PRED_6_VAL <- predict(Model6, validation)
(RMSE_6_IN<-sqrt(sum((PRED_6_IN-train$gross)^2)/length(PRED_6_IN))) #computes in-sample error
(RMSE_6_OUT<-sqrt(sum((PRED_6_OUT-test$gross)^2)/length(PRED_6_OUT)))
(RMSE_6_VAL<-sqrt(sum((PRED_6_VAL-validation$gross)^2)/length(PRED_6_VAL)))
#non-linear transformations
Model7 <- lm(gross ~ poly(budget, degree = 3) + genre + score + votes + runtime, data = train) #rating
summary(Model7)

PRED_7_IN <- predict(Model7, train)
PRED_7_OUT <- predict(Model7, test)
PRED_7_VAL <- predict(Model7, validation)
(RMSE_7_IN <- sqrt(sum((PRED_7_IN - train$gross)^2) / length(PRED_7_IN)))  # in-sample error
(RMSE_7_OUT <- sqrt(sum((PRED_7_OUT - test$gross)^2) / length(PRED_7_OUT)))
(RMSE_7_VAL <- sqrt(sum((PRED_7_VAL - validation$gross)^2) / length(PRED_7_VAL)))




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
PRED_8_Val <- predict(SVM_Model, validation)
(RMSE_8_IN <- sqrt(sum((PRED_8_IN - train$gross)^2) / length(PRED_8_IN)))  # in-sample error
(RMSE_8_OUT <- sqrt(sum((PRED_8_OUT - test$gross)^2) / length(PRED_8_OUT)))
(RMSE_8_VAL <- sqrt(sum((PRED_8_Val - validation$gross)^2) / length(PRED_8_Val)))

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
print(reg_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
rg_fmla <- gross ~ rating + genre + score + votes + runtime + budget
reg_tree <- reg_spec %>%
  fit(formula = rg_fmla, data = train)
print(reg_tree)

#VISUALIZING THE CLASSIFICATION TREE MODEL:
reg_tree$fit %>%
  rpart.plot(type = 4, roundint = FALSE)

plotcp(reg_tree$fit)


pred_10_in <- predict(reg_tree, new_data = train) %>%
  bind_cols(train)
pred_10_out <- predict(reg_tree, new_data = test) %>%
  bind_cols(test)
pred_10_val <- predict(reg_tree, new_data = validation) %>%
  bind_cols(validation)

RMSE_10_Out <- rmse(pred_10_out, estimate=.pred, truth=gross)
RMSE_10_In <- rmse(pred_10_in, estimate=.pred, truth=gross)
RMSE_10_In <- RMSE_10_In$.estimate
RMSE_10_Out <- RMSE_10_Out$.estimate
RMSE_10_VAL <- rmse(pred_10_val, estimate=.pred, truth=gross)
RMSE_10_VAL <- RMSE_10_VAL$.estimate

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
set.seed(123)
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
pred_12_val <- predict(bagged_forest, new_data = validation) %>%
  bind_cols(validation)

RMSE_12_Out <- rmse(pred_12_out, estimate=.pred, truth=gross)
RMSE_12_In <- rmse(pred_12_in, estimate=.pred, truth=gross)
RMSE_12_In <-RMSE_12_In$.estimate
RMSE_12_Out <- RMSE_12_Out$.estimate

RMSE_12_VAL <- rmse(pred_12_val, estimate=.pred, truth=gross)
RMSE_12_VAL <- RMSE_12_VAL$.estimate

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

tune_results
#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_params <- select_best(tune_results)

#FINALIZE THE MODEL SPECIFICATION
final_spec <- finalize_model(tree_spec, best_params)

final_spec


#Table#

TABLE_MultiREG<- as.table(matrix(c(RMSE_5_IN,RMSE_6_IN,RMSE_7_IN,RMSE_8_IN,RMSE_10_In,RMSE_12_In,RMSE_5_OUT,RMSE_6_OUT,RMSE_7_OUT,RMSE_8_OUT,RMSE_10_Out,RMSE_12_Out), ncol=6, byrow=TRUE))
colnames(TABLE_MultiREG) <- c('Model Linear','Model Regularized', 'Model Non-Linear Trans', "Model SVM", "Model RegressionTree", "Model Bagged Tree")
rownames(TABLE_MultiREG) <- c('RMSE_In','RMSE_Out')
TABLE_MultiREG

#Validation#
TABLE_MultiREGVAL<- as.table(matrix(c(RMSE_5_VAL,RMSE_6_VAL,RMSE_7_VAL,RMSE_8_VAL,RMSE_10_VAL,RMSE_12_VAL), ncol=6, byrow=TRUE))
colnames(TABLE_MultiREGVAL) <- c('Model Linear','Model Regularized', 'Model Non-Linear Trans', "Model SVM", "Model RegressionTree", "Model Bagged Tree")
rownames(TABLE_MultiREGVAL) <- c('RMSE_VAL')
TABLE_MultiREGVAL


help("cbind")
library(dplyr)
oscars_df <- mutate(oscars_df, Winner = ifelse(oscars_df$Award == "Winner",1,0)) 
colnames(movies_1)[colnames(movies_1) == "name"] <- "Film"                        
merged_df <- left_join(oscars_df,movies_1, by = "Film")                          
merged_2 <- subset(merged_df, !is.na(budget))
merged_3 <- na.omit(merged_df)
Oscars<- merged_2








#Classification#
set.seed(111)
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

confusion1_in<-table(Predictions_13_In, train2$Winner == 1)
confusion

confusion1_out<-table(Predictions_13_Out, test2$Winner == 1)

confusionMatrix(confusion1_in, positive='TRUE')
confusionMatrix(table(Predictions_13_In >= 0.5, train2$Winner == 1), positive='TRUE')

confusionMatrix(confusion_out, positive='TRUE')
confusionMatrix(table(Predictions_13_Out >= 0.5, test2$Winner == 1), positive='TRUE')

Confusion13_in <- confusionMatrix(table(Predictions_13_In >= 0.5, train2$Winner == 1), positive='TRUE') 
Accuracy_13_in <- Confusion13_in$overall['Accuracy']
Confusion13_out <- confusionMatrix(table(Predictions_13_Out >= 0.5, test2$Winner == 1), positive='TRUE') 
Accuracy_13_out <- Confusion13_out$overall['Accuracy']
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
(E_IN_RETxUNE<-1-mean(predict(SVM_Retune15, train2)==train2$Winner))
(E_OUT_RETUNE<-1-mean(predict(SVM_Retune15, test2)==test2$Winner))

Accuracy_15_in <- 1-(E_IN_RETUNE<-1-mean(predict(SVM_Retune15, train2)==train2$Winner))
Accuracy_15_in
Accuracy_15_out <- 1-(E_OUT_RETUNE<-1-mean(predict(SVM_Retune15, test2)==test2$Winner))
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
  bind_cols(train2)
confusion_16 <- table(pred_class16$.pred_class, pred_class16$Winner)
confusionMatrix(confusion_16, positive = "1")


pred_class16out <- predict(class_tree16, new_data = test2, type="class") %>%
  bind_cols(test2) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

pred_prob16out <- predict(class_tree16, new_data = test2, type="prob") %>%
  bind_cols(test2)
confusion_16_out <- table(pred_class16out$.pred_class, pred_class16out$Winner)
confusionMatrix(confusion, positive = "1")



#Tuning Decision Tree#
set.seed(111)
split<-initial_split(Oscars, prop=.7)
train2<-training(split)
holdout2<-testing(split)

split_2 <- initial_split(holdout2, prop = .5)
test2 <- training(split_2)
validation2 <- testing(split_2)

Tun16 <- Winner ~ gross + votes + runtime + score
tree_spec <- decision_tree(min_n = tune(),
                           tree_depth = tune(),
                           cost_complexity= tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")
tree_grid <- grid_regular(parameters(tree_spec), levels = 3)
tune_results <- tune_grid(tree_spec,
                          Tun16, #MODEL FORMULA
                          resamples = vfold_cv(train2, v=3), #RESAMPLES / FOLDS
                          grid = tree_grid, #GRID
                          metrics = metric_set(rmse)) #BENCHMARK METRIC
#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_params <- select_best(tune_results)

#FINALIZE THE MODEL SPECIFICATION
final_spec <- finalize_model(tree_spec, best_params)
final_spec

class_spec_final <- decision_tree(min_n = best_params$min_n , #minimum number of observations for split
                            tree_depth =  best_params$tree_depth, #max tree depth
                            cost_complexity = best_params$cost_complexity)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec)

class_16_tune <- factor(Winner) ~ gross + votes + runtime + score
class_tree16_tune <- class_spec_final %>%
  fit(formula = class_16_tune, data = train2)
print(class_tree16_tune)

class_tree16_tune$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

plotcp(class_tree16_tune$fit)

pred_class16_tune <- predict(class_tree16_tune, new_data = train2, type="class") %>%
  bind_cols(train2)

pred_class16_tune_out <- predict(class_tree16_tune, new_data = test2, type="class") %>%
  bind_cols(test2)

confusion_16_tune <- table(pred_class16_tune$.pred_class, pred_class16_tune$Winner)
confusionMatrix(confusion_16_tune, positive = "1")

confusion_16_tune_out <- table(pred_class16_tune_out$.pred_class, pred_class16_tune_out$Winner)
confusionMatrix(confusion_16_tune_out, positive = "1")



Confusion_16_in <- confusionMatrix(confusion_16, positive = "1")
Accuracy_16_in <- Confusion_16_in$overall['Accuracy']
Confusion_16_out <- confusionMatrix(confusion_16_out, positive = "1")
Accuracy_16_out <- Confusion_16_out$overall['Accuracy']
#Gradient Boosted#
Model_ <- factor(Winner) ~ budget + gross + votes + score + runtime
boosted_forest <- boost_tree(min_n = NULL, #minimum number of observations for split
                            tree_depth = NULL, #max tree depth
                            trees = 100, #number of trees
                            mtry = NULL, #number of predictors selected at each split 
                            sample_size = NULL, #amount of data exposed to fitting
                            learn_rate = NULL, #learning rate for gradient descent
                            loss_reduction = NULL, #min loss reduction for further split
                            stop_iter = NULL)  %>% #maximum iteration for convergence
  set_engine("xgboost") %>%
  set_mode("classification") %>%
  fit(Model_, train2)



#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_xb_in <- predict(boosted_forest, new_data = train2, type="class") %>%
  bind_cols(train2) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_17 <- table(pred_class_xb_in$.pred_class, pred_class_xb_in$Winner)
confusionMatrix(confusion_17) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_xb_out <- predict(boosted_forest, new_data = test2, type="class") %>%
  bind_cols(test2) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_17_Out <- table(pred_class_xb_out$.pred_class, pred_class_xb_out$Winner)
confusionMatrix(confusion_17_Out) #FROM CARET PACKAGE

Confusion_17_in <- confusionMatrix(confusion_17, positive = "1")
Accuracy_17_in <- Confusion_17_in$overall['Accuracy']
confusion_17out <- confusionMatrix(confusion_17_Out, positive = "1")
Accuracy_17_out <- confusion_17out$overall['Accuracy']


#Tuned#

boost_spec <- boost_tree(
  trees = 500,
  learn_rate = tune(),
  tree_depth = tune(),
  sample_size = tune()) %>%
  set_mode("classification") %>%
  set_engine("xgboost") 
  
tunegrid_boost <- grid_regular(parameters(boost_spec), levels = 2)

tune_results_boost <- tune_grid(
  boost_spec,
  Model_,
  resamples = vfold_cv(train2, v = 6),
  grid = tunegrid_boost,
  metrics = metric_set(accuracy)
)

best_params_boost <- select_best(tune_results_boost)
final_spec_boost <- finalize_model(boost_spec, best_params_boost)

final_model_boost <- final_spec_boost %>% fit(Model_, data = train2)

pred_class_xb_outtune <- predict(final_model_boost, new_data = test2, type="class") %>%
  bind_cols(test2) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_17_Outtune <- table(pred_class_xb_outtune$.pred_class, pred_class_xb_outtune$Winner)
confusionMatrix(confusion_17_Outtune)


#Table#
TABLE_Class <- as.table(matrix(c(Accuracy_13_in,Accuracy_15_in,Accuracy_16_in,Accuracy_17_in,Accuracy_13_out,Accuracy_15_out,Accuracy_16_out,Accuracy_17_out), ncol=4, byrow=TRUE))
colnames(TABLE_Class) <- c('Model 13','Model 15', 'Model 16', 'Model 17')
rownames(TABLE_Class) <- c('Accuracy_IN', 'Accuracy_OUT')
TABLE_Class

#Validation#
Confusion_1_Val<- confusionMatrix(table(predict(Model_13, validation2, type = "response") >= 0.5, validation2$Winner == 1), positive='TRUE')
Accuracy_1_Val <- Confusion_1_Val$overall['Accuracy']
Accuracy_2_Val <- 1-(1-mean(predict(SVM_Retune15, validation2)==validation2$Winner))
pred_class16val <- predict(class_tree16, new_data = validation2, type="class") %>%
  bind_cols(validation2) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA
pred_prob16val <- predict(class_tree16, new_data = validation2, type="prob") %>%
  bind_cols(validation2)
confusion_16_val <- table(pred_class16val$.pred_class, pred_class16val$Winner)
Confusion16val <- confusionMatrix(confusion_16_val, positive = "1")
Accuracy_3_Val <- Confusion16val$overall['Accuracy']


#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_xb_val <- predict(boosted_forest, new_data = validation2, type="class") %>%
  bind_cols(validation2) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_17_Val <- table(pred_class_xb_val$.pred_class, pred_class_xb_val$Winner)
CONFUSION_17Val <- confusionMatrix(confusion_17_Val) #FROM CARET PACKAGE
Accuracy_4_Val <- CONFUSION_17Val$overall['Accuracy']
confusionMatrix(confusion_17_Val)


TABLE_Val_Binary <- as.table(matrix(c(Accuracy_1_Val,Accuracy_2_Val,Accuracy_3_Val,Accuracy_4_Val), ncol=4, byrow=TRUE))
colnames(TABLE_Val_Binary) <- c('Model 13','Model 15', 'Model 16', 'Model 17')
rownames(TABLE_Val_Binary) <- c('Accuracy_Val')
TABLE_Val_Binary

#Model 4 is best#
str(movies_1)


#Multi-Class#

movies_2 <- movies_1 %>%
  mutate(rating = case_when(
    rating == "PG" ~ 1,
    rating == "PG-13" ~ 2,
    rating == "R" ~ 3,
    TRUE ~ 0  # For everything else
  ))

set.seed(120)
split_3<-initial_split(movies_2, prop=.7)
train3<-training(split_3)
holdout3<-testing(split_3)

split_4 <- initial_split(holdout3, prop = .5)
test3 <- training(split_4)
validation3 <- testing(split_4)


kern_type<-"radial" #SPECIFY KERNEL TYPE

#BUILD SVM CLASSIFIER
SVM_MultiClass<- svm(rating ~ gross + budget + score + runtime + votes, 
                data = train3, 
                type = "C-classification", #set to "eps-regression" for numeric prediction
                kernel = kern_type,
                cost=1,                   #REGULARIZATION PARAMETER
                gamma = 1/(ncol(training)-1), #DEFAULT KERNEL PARAMETER
                coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                scale = FALSE)  
print(SVM_MultiClass)
(E_IN_MultiClass<-1-mean(predict(SVM_MultiClass, train3)==train3$rating))
(E_OUT_MultiClass<-1-mean(predict(SVM_MultiClass, test3)==test3$rating))

Accuracy_1_in_mutli <- 1 -(E_IN_MultiClass<-1-mean(predict(SVM_MultiClass, train3)==train3$rating))
Accuracy_1_out_mutli <- 1- (E_OUT_MultiClass<-1-mean(predict(SVM_MultiClass, test3)==test3$rating))
Accuracy_1_in_mutli
Accuracy_1_out_mutli

#Tune SVM Model#
tune_control<-tune.control(cross=5) #SET K-FOLD CV PARAMETERS
TUNE_Class <- tune.svm(x = train3[,c(6,7,12,13)],
                 y = factor(train3$rating),
                 type = "C-classification",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(.01, .1, 1, 10, 100, 1000), #REGULARIZATION PARAMETER
                 gamma = 1/(ncol(train3)-1), #KERNEL PARAMETER
                 coef0 = 0,           #KERNEL PARAMETER
                 degree = 2)          #POLYNOMIAL KERNEL PARAMETER


SVM_MultiClass_tune<- svm(rating ~ gross + budget + score + runtime + votes, 
                     data = train3, 
                     type = "C-classification", #set to "eps-regression" for numeric prediction
                     kernel = kern_type,
                     cost= TUNE_Class$best.parameters$cost,                   #REGULARIZATION PARAMETER
                     gamma = TUNE_Class$best.parameters$gamma, #DEFAULT KERNEL PARAMETER
                     coef0 = TUNE_Class$best.parameters$coef0,                    #DEFAULT KERNEL PARAMETER
                     degree=TUNE_Class$best.parameters$degree,                     #POLYNOMIAL KERNEL PARAMETER
                     scale = FALSE)  

print(SVM_MultiClass_tune)
1-(E_IN_MultiClass<-1-mean(predict(SVM_MultiClass_tune, train3)==train3$rating))
1-(E_OUT_MultiClass<-1-mean(predict(SVM_MultiClass_tune, test3)==test3$rating))



#Class Tree#
library(rpart.plot)

Multi_tree <- rpart(factor(rating) ~ gross + votes + runtime + score, data = train3, method = "class")


multiclass_spec <- decision_tree(min_n = 20 , #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec)
str(train)

multi_class_m1<- factor(rating) ~ gross + votes + runtime + score
Multi_class_tree <- multiclass_spec%>%
  fit(formula = factor(rating) ~ gross + votes + runtime + score, data = train3)
print(Multi_class_tree)

Multi_class_tree$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

plotcp(Multi_class_tree$fit)

pred_multiclass <- predict(Multi_class_tree, new_data = train3, type="class") %>%
  bind_cols(train3) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA


confusion_multi <- table(pred_multiclass$.pred_class, pred_multiclass$rating)
Confusionmatrix2_in_multi <- confusionMatrix(confusion_multi, positive = "1")
Confusionmatrix2_in_multi

pred_multiclass_out <- predict(Multi_class_tree, new_data = test3, type="class") %>%
  bind_cols(test3)

confusion_multi_out <- table(pred_multiclass_out$.pred_class, pred_multiclass_out$rating)
Confusionmatrix2_out_multi<- confusionMatrix(confusion_multi_out, positive = "1")
Confusionmatrix2_out_multi

Accuracy_2_in_mutli <- Confusionmatrix2_in_multi$overall['Accuracy']
Accuracy_2_out_mutli <- Confusionmatrix2_out_multi$overall['Accuracy']
Accuracy_2_in_mutli
Accuracy_2_out_mutli


#Tuneing of Tree#
tree_spec_Multi <- decision_tree(min_n = tune(),
                           tree_depth = tune(),
                           cost_complexity= tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

tree_grid_Multi <- grid_regular(parameters(tree_spec_Multi), levels = 3)

set.seed(123) #SET SEED FOR REPRODUCIBILITY WITH CROSS-VALIDATION
tune_results_multi <- tune_grid(tree_spec_Multi,
                          factor(rating) ~ gross + votes + runtime + score, #MODEL FORMULA
                          resamples = vfold_cv(train3, v=3), #RESAMPLES / FOLDS
                          grid = tree_grid_Multi, #GRID
                          metrics = metric_set(accuracy)) #BENCHMARK METRIC

#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_params_multi <- select_best(tune_results_multi)

#FINALIZE THE MODEL SPECIFICATION
final_spec_multi <- finalize_model(tree_spec, best_params_multi)
final_spec_multi

Multi_tree <- rpart(factor(rating) ~ gross + votes + runtime + score, data = train3, method = "class")


multiclass_spec_tune <- decision_tree(min_n = best_params_multi$min_n , #minimum number of observations for split
                                 tree_depth = best_params_multi$tree_depth, #max tree depth
                                 cost_complexity = best_params_multi$cost_complexity )  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(multiclass_spec_tune)
str(train)

multi_class_tune<- factor(rating) ~ gross + votes + runtime + score
Multi_class_tree_tune <- multiclass_spec_tune%>%
  fit(formula = factor(rating) ~ gross + votes + runtime + score, data = train3)
print(Multi_class_tree_tune)

Multi_class_tree_tune$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

plotcp(Multi_class_tree_tune$fit)


pred_multiclassfinal <- predict(Multi_class_tree_tune, new_data = train3, type="class") %>%
  bind_cols(train3) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

pred_multiclassfinalout <- predict(Multi_class_tree_tune, new_data = test3, type="class") %>%
  bind_cols(test3) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA



confusion_multifinal <- table(pred_multiclassfinal$.pred_class, pred_multiclass$rating)
ConfusionmatrixFinal_in_multi <- confusionMatrix(confusion_multi, positive = "1")
ConfusionmatrixFinal_in_multi

confusion_multifinalout <- table(pred_multiclassfinalout$.pred_class, pred_multiclass_out$rating)
ConfusionmatrixFinal_out_multi <- confusionMatrix(confusion_multi_out, positive = "1")
ConfusionmatrixFinal_out_multi



#Random Forest#
spec_rf_multi <- rand_forest(min_n = 20 , #minimum number of observations for split
                       trees = 100, #of ensemble members (trees in forest)
                       mtry = 2)  %>% #number of variables to consider at each split
  set_mode("classification") %>% #can set to regression for numeric prediction
  set_engine("ranger") #alternative engine / package: randomForest
spec_rf_multi

#FITTING THE RF MODEL
set.seed(123) #NEED TO SET SEED WHEN FITTING OR BOOTSTRAPPED SAMPLES WILL CHANGE
random_forest_multi <- spec_rf_multi %>%
  fit(formula = factor(rating) ~ gross + budget + score + votes, data = train3) #%>%
print(random_forest)

#RANKING VARIABLE IMPORTANCE (CAN BE DONE WITH OTHER MODELS AS WELL)
set.seed(123) #NEED TO SET SEED WHEN FITTING OR BOOTSTRAPPED SAMPLES WILL CHANGE
rand_forest(min_n = 20 , #minimum number of observations for split
            trees = 100, #of ensemble members (trees in forest)
            mtry = 2)  %>% #number of variables to consider at each split
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity") %>%
  fit(formula = factor(rating) ~ gross + budget + score + votes, data = train3) %>%
  vip() #FROM VIP PACKAGE - ONLY WORKS ON RANGER FIT DIRECTLY

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_rf_in_multi <- predict(random_forest_multi, new_data = train3, type="class") %>%
  bind_cols(train3) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA


#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_in_multi <- table(pred_class_rf_in_multi$.pred_class, pred_class_rf_in_multi$rating)
Matrix_3_in <- confusionMatrix(confusion_in_multi) #FROM CARET PACKAGE
Matrix_3_in
#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_rf_out_multi <- predict(random_forest_multi, new_data = test3, type="class") %>%
  bind_cols(test3)


#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_out_multi <- table(pred_class_rf_out_multi$.pred_class, pred_class_rf_out_multi$rating)
ConfusionMatrix_3_Out <- confusionMatrix(confusion_out_multi) #FROM CARET PACKAGE
ConfusionMatrix_3_Out

Accuracy_3_in_multi <- Matrix_3_in$overall["Accuracy"]
Accuracy_3_out_multi <- ConfusionMatrix_3_Out$overall["Accuracy"]

Accuracy_3_in_multi

#Tuneing Random FOrest#
library(ranger)
library(e1071)

hyperparameters <- list(
  num.trees = c(50, 100, 150),
  mtry = c(2, 4, 6))

help("tune.randomForest")


tuned_model <- tune.randomForest(
  x = train3[,c(6,7,12,13)],  # Exclude the target variable
  y = train3$rating,
  ntree = hyperparameters$num.trees,
  mtryStart = min(hyperparameters$mtry))


set.seed(123)
rand_forest(min_n = c(20,30,40,50,60,70) , #minimum number of observations for split
            trees = 500, #of ensemble members (trees in forest)
            mtry = c(2,3,4,5,6)) %>%
set_mode("classification") %>%
  set_engine("ranger", importance = "impurity") %>%
  fit(formula = factor(rating) ~ gross + budget + score + votes, data = train3) %>%
  vip() #FROM VIP PACKAGE - ONLY WORKS ON RANGER FIT DIRECTLY



#Table Multi Class#
TABLE_MultiClass <- as.table(matrix(c(Accuracy_1_in_mutli,Accuracy_2_in_mutli,Accuracy_3_in_multi,Accuracy_1_out_mutli,Accuracy_2_out_mutli,Accuracy_3_out_multi), ncol=3, byrow=TRUE))
colnames(TABLE_MultiClass) <- c('Model 1','Model 2', 'Model 3')
rownames(TABLE_MultiClass) <- c('Accuracy_IN', 'Accuracy_OUT')
TABLE_MultiClass

#Validation#
Accuracy_1_ValMulti <- 1-(1-mean(predict(SVM_MultiClass, validation3)==validation3$rating))
pred_class2val <- predict(Multi_class_tree, new_data = validation3, type="class") %>%
  bind_cols(validation3) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA
confusion_2_val <- table(pred_class2val$.pred_class, pred_class2val$rating)
Confusion2val <- confusionMatrix(confusion_2_val, positive = "1")
Accuracy_2_ValMulti <- Confusion2val$overall['Accuracy']
pred_class3val <- predict(random_forest_multi, new_data = validation3, type = "class") %>%
  bind_cols(validation3)
confusion_3_val <- table(pred_class3val$.pred_class, pred_class3val$rating)
Confusion3val <- confusionMatrix(confusion_3_val, positive = "1")
Accuracy_3_ValMulti <- Confusion3val$overall["Accuracy"]

TABLE_MultiVal<- as.table(matrix(c(Accuracy_1_ValMulti,Accuracy_2_ValMulti,Accuracy_3_ValMulti), ncol=3, byrow=TRUE))
colnames(TABLE_MultiVal) <- c('Model 1','Model 2', 'Model 3')
rownames(TABLE_MultiVal) <- c('Accuracy_Val')
TABLE_MultiVal

Confusion3val
