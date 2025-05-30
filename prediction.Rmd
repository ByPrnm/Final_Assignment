source("finalProject.r")
# Load required libraries
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)

# Read the data
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

# Data preprocessing
# Remove columns with mostly NA values
na_cols <- colMeans(is.na(training)) > 0.95
training <- training[, !na_cols]
testing <- testing[, !na_cols]

# Remove near zero variance predictors
nzv <- nearZeroVar(training)
training <- training[, -nzv]
testing <- testing[, -nzv]

# Remove identification and timestamp columns
training <- training[, -(1:7)]
testing <- testing[, -(1:7)]

# Split training data into training and validation sets
set.seed(12345)
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
train_data <- training[inTrain, ]
validation_data <- training[-inTrain, ]

# Train Random Forest model with cross-validation
ctrl <- trainControl(method = "cv", number = 5)
rf_model <- train(classe ~ ., data = train_data, method = "rf",
                 trControl = ctrl, ntree = 500)

# Make predictions on validation set
val_pred <- predict(rf_model, validation_data)
confusion_matrix <- confusionMatrix(val_pred, factor(validation_data$classe))

# Print model performance
print("Model Performance on Validation Set:")
print(confusion_matrix)

# Calculate out of sample error
oos_error <- 1 - confusion_matrix$overall["Accuracy"]
print(paste("Out of Sample Error Rate:", round(oos_error * 100, 2), "%"))

# Feature importance
importance <- varImp(rf_model)
print("Top 10 Most Important Features:")
print(importance$importance[1:10, ])

# Create visualization of feature importance
plot(importance, top = 20, main = "Top 20 Important Features")

# Make predictions on test set
test_pred <- predict(rf_model, testing)
print("Predictions for Test Cases:")
print(test_pred)

# Save predictions to file
predictions <- data.frame(problem_id = testing$problem_id, predicted = test_pred)
write.csv(predictions, "predictions.csv", row.names = FALSE)
