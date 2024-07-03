# Getting Required Libraries
library('ggplot2')
library('dplyr')
library('reshape2')
library('caret')
library('e1071')
library('glue')
library('ModelMetrics')
library('rpart')
library('ROSE')
library('smotefamily')
library('DMwR2')
library('deforestable')
library('randomForest')

#Loading The Data
data <- data.table::fread("D:\\college\\SY\\SY COMMON\\DS\\CP\\creditcard.csv")

table(data$Class)
predictors <- select(data, -Class)

 cbind(melt(apply(predictors, 2, min), value.name = "min"), 
      melt(apply(predictors, 2, max), value.name = "max"))

predictors_rescaled <- as.data.frame(apply(predictors, 2, scale ))

cbind(melt(apply(predictors_rescaled, 2, min), value.name = "min_after_rescaling"), 
      melt(apply(predictors_rescaled, 2, max), value.name = "max_after_rescaling"))

data <- cbind(Class = data$Class, predictors_rescaled)
set.seed(23)

sample <- sample_n(data, 10000)

sample_smote <- SMOTE(
  X = sample[, -1],
  target = sample$Class,
  dup_size = 4
)

sample_smote_data <- sample_smote$data
sample_smote_data$class <- factor(sample_smote_data$class)
levels(sample_smote_data$class)
table(sample_smote_data$class)

sample_smote_under <- ovun.sample(class ~ .,
                                  data = sample_smote_data,
                                  method = "under",
                                  N = nrow(sample_smote_data[sample_smote_data$class == 1, ]) * 11
)

sample_smote_under_data <- sample_smote_under$data
levels(sample_smote_under_data$class)
sample_smote_under_data$class <- relevel(sample_smote_under_data$class, ref = 1)

# visualization
p1 <- ggplot(sample, aes(x = V1, y = V2, color = factor(Class))) +
  geom_point(alpha = 0.3) +
  facet_wrap(~Class, labeller = labeller(Class = c("1" = "Fraud", "0" = "Not Fraud"))) +
  labs(
    title = "Before SMOTE",
    subtitle = "For 10,000 Random Sample",
    color = "Class"
  ) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("blue", "red")) +
  theme(legend.position = "none")
p1
p2 <- ggplot(sample_smote_under_data, aes(x = V1, y = V2, color = factor(class))) +
  geom_point(alpha = 0.3) +
  facet_wrap(~class, labeller = labeller(class = c("1" = "Fraud", "0" = "Not Fraud"))) +
  labs(
    title = "After SMOTE & Random Majority Undersampling",
    subtitle = "with 10:1 majority:minority ratio",
    color = "Class"
  ) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("red", "blue")) +
  theme(legend.position = "none")

p2

sample$Class<-as.factor(sample$Class)
train_index <- sample(seq_len(nrow(sample)), size = 0.75*nrow(sample))

train <- sample[train_index, ] # training data (75% of data)
test <- sample[-train_index, ] # testing data (25% of data)

# under-sample until majority sample size matches
#For First Model with 50-50 balance
smote_v1 <- SMOTE(X = train[, -1], target = train$Class, dup_size = 29)
smote_train_v1 <- smote_v1$data %>% rename(Class = class)

under_v1 <- ovun.sample(Class ~ .,
                        data = smote_train_v1,
                        method = "under",
                        N = 2 * sum(smote_train_v1$Class == 1)
)

train_v1 <- under_v1$data
#For second Model with 75-25 balance
smote_v2 <- SMOTE(X = train[, -1], target = train$Class, dup_size = 29)
smote_train_v2 <- smote_v2$data %>% rename(Class = class)

under_v2 <- ovun.sample(Class ~ .,
                        data = smote_train_v2,
                        method = "under",
                        N = (sum(smote_train_v2$Class == 1) * 4)
)

train_v2 <- under_v2$data



table(train$Class)

rf.model <- randomForest(Class ~ ., data = train, 
                         ntree = 500,
                         mtry = 5)
print(rf.model)


rf.predict <- predict(rf.model, test)

test$Class <- as.factor(test$Class)

confusionMatrix(test$Class, rf.predict)

#Larger, Balanced (50:50) – train_v1

table(train_v1$Class)
train_v1$Class <- as.numeric(train_v1$Class)

rf.model_v1 <- randomForest(Class ~ ., data = train_v1, 
                            ntrees = 500,
                            mtry = 5)

rf.predict_v1 <- predict(rf.model_v1, test)

confusionMatrix(test$Class, rf.predict_v1)
print(rf.model_v1)

#Larger, Fraud-Minority (25:75) – train_v2

table(train_v2$Class)
train_v2$Class <- as.numeric(train_v2$Class)

rf.model_v2 <- randomForest(Class ~ ., data = train_v2, 
                            ntrees = 500,
                            mtry = 5)

rf.predict_v2 <- predict(rf.model_v2, test)
rf.predict_v2
test$Class <- as.numeric(test$Class)
confusionMatrix(test$Class, rf.predict_v2)
confusionMatrix(predict(rf.model_v2,test),test$Class)

print(rf.model_v2)

# Fraud Detection

library(readr)
library(plyr)
library(caret)
library(rpart)
library(mboost)
library(MASS)
library(pamr)
library(dplyr)
library(naivebayes)

Credit_Card_Fraud_Detection <- read.csv("D:\\college\\SY\\SY COMMON\\DS\\CP\\creditcard.csv")

names(Credit_Card_Fraud_Detection)[names(Credit_Card_Fraud_Detection)=="Class"] <- "Fraud"

Credit_Card_Fraud_Detection$Fraud <- factor(Credit_Card_Fraud_Detection$Fraud, labels = c("Normal", "Fraud"))
Credit_Card_Fraud_Detection$Fraud <- relevel(Credit_Card_Fraud_Detection$Fraud, "Fraud")

any(is.na(Credit_Card_Fraud_Detection))

set.seed(1, sample.kind = "Rounding") # just for make the code reproducible

validation_index <- createDataPartition(y = Credit_Card_Fraud_Detection$Fraud, times = 1, p = 0.2, list = FALSE)

validation <- Credit_Card_Fraud_Detection[validation_index,]
df <- Credit_Card_Fraud_Detection[-validation_index,]

train_index <- createDataPartition(y = df$Fraud, times = 1, p = 0.8, list = FALSE)

train_set <- df[train_index,]
test_set <- df[-train_index,]

str(df[,c(1,2,3,30,31)])
summary(df[,c(1,2,3,30,31)])


df %>%
  group_by(Fraud) %>%
  ggplot() +
  geom_density(aes(Amount, fill = Fraud), alpha = .5) +
  scale_x_log10() +
  scale_fill_manual(values = c("#E69F00", "#0072B2"))

fit_bad_model <- train(Fraud~.,
                       data = train_set,
                       method = "rpart",
                       trControl = trainControl(method = "cv", number = 5))

hat_bad_model <- predict(fit_bad_model, newdata = test_set)

confusionMatrix(data = hat_bad_model, reference = test_set$Fraud)


df_trainup <- upSample(y=df$Fraud,
                       x=df[,-ncol(df)],
                       yname = "Fraud")

table(df_trainup$Fraud)


fit_rpart_final <- train(Fraud ~ .,
                         data = df_trainup,
                         method = "rpart",
                         trControl = trainControl(method = "cv", number = 5),
                         preProcess = c("center", "scale"))

p_rpart_final <- predict(fit_rpart_final, newdata = validation, type = "prob")


fit_glm_final <- train(Fraud ~ .,
                       data = df_trainup,
                       method = "glm",
                       trControl = trainControl(method = "cv", number = 5),
                       preProcess = c("center", "scale"))

p_glm_final <- predict(fit_glm_final, newdata = validation, type = "prob")


fit_pam_final <- train(Fraud ~ .,
                       data = df_trainup,
                       method = "pam",
                       trControl = trainControl(method = "cv", number = 5),
                       preProcess = c("center", "scale"))

p_pam_final <- predict(fit_pam_final, newdata = validation, type = "prob")


fit_nb_final <- train(Fraud ~ .,
                      data = df_trainup,
                      method = "naive_bayes",
                      trControl = trainControl(method = "cv", number = 5),
                      preProcess = c("center", "scale"))

p_nb_final <- predict(fit_nb_final, newdata = validation, type = "prob")


fit_glmboost_final <- train(Fraud ~ .,
                            data = df_trainup,
                            method = "glmboost",
                            trControl = trainControl(method = "cv", number = 5),
                            preProcess = c("center", "scale"))

p_glmboost_final <- predict(fit_glmboost_final, newdata = validation, type = "prob")


fit_lda_final <- train(Fraud ~ .,
                       data = df_trainup,
                       method = "lda",
                       trControl = trainControl(method = "cv", number = 5),
                       preProcess = c("center", "scale"))

p_lda_final <- predict(fit_lda_final, newdata = validation, type = "prob")


p_final <- (p_rpart_final + p_glm_final + p_pam_final + p_nb_final + p_glmboost_final + p_lda_final)/6

y_final_pred <- factor(apply(p_final, 1, which.max))

y_final_pred <- factor(y_final_pred, labels = c("Fraud", "Normal"))


confusionMatrix(y_final_pred, validation$Fraud)
