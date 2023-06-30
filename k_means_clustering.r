dataset <- read.csv("diabetes_prediction_dataset.csv")
summary(dataset)

#Data Cleaning
#Rounding the 'age' variable to integer
dataset$age <- round(dataset$age)
#Convert Male to '1' and Female to '0'
dataset$gender <- ifelse(dataset$gender == "Male", 1, 0)
#Removing rows where the value of 'smoking_history' variable is "No Info".
dataset <- subset(dataset, smoking_history != "No Info")
#Creating an ordered factor with levels as "current", "ever", "former", "never", "not current".
ordinal_scale <- c("current", "ever", "former", "never", "not current")
numeric_values <- c(1, 2, 3, 4, 5)
dataset$smoking_history <- as.integer(factor(dataset$smoking_history, levels = ordinal_scale))
dataset$smoking_history<- numeric_values[dataset$smoking_history]
summary(dataset)

#Represents the correlation coefficients between the "diabetes" variable and the other factors
library(ggplot2)
correlation_matrix <- cor(dataset)
# Extract the correlation coefficients between each factor and diabetes
correlation_diabetes <- correlation_matrix[, "diabetes"]
# Sort the correlation coefficients in descending order
correlation_sorted <- sort(abs(correlation_diabetes), decreasing = TRUE)
# Print the correlation coefficients and corresponding factors
correlation_result <- data.frame(Factor = names(correlation_sorted), Correlation = correlation_sorted)
correlation_result
#Plot the correlation coefficients
ggplot(data = correlation_result, aes(x = Factor, y = Correlation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Factor", y = "Correlation") +
  ggtitle("Correlation Coefficients with Diabetes") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Setting the seed value to ensure some output everytime
#Don't change number of sample as it also will change the cluster representation which will cause the error to confusion matrix  
library(dplyr)
set.seed(100)
#Select 100 data points with diabetes = '0'
diabetes_0_samples <- dataset %>%
  filter(diabetes == '0') %>%
  sample_n(100)    
#Select 100 data points with diabetes = '1'
diabetes_1_samples <- dataset %>%
  filter(diabetes == '1') %>%
  sample_n(100)
#Combine the two datasets
diabetes_data <- bind_rows(diabetes_0_samples, diabetes_1_samples)
head(diabetes_data)

#Based on the correlation coefficients, we drop the less relevant column:gender and smoking history
#We also exclude the "diabetes" column as it does not contribute to the unsupervised k-means clustering process.
sample_data <- diabetes_data[, c("age", "hypertension", "heart_disease","bmi", "HbA1c_level", "blood_glucose_level")]

#To determine the optimal number of cluster
library(factoextra)
# Elbow method
fviz_nbclust(sample_data, kmeans, method='wss')+ labs(subtitle= 'Elbow method')
# Silhouette method
fviz_nbclust(sample_data, kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette method")

# Scale the sample_data to standardize the variables.
sample_data_scale <- scale(sample_data)
# Perform k-means clustering with 2 centers and 100 random starts.
model <- kmeans(sample_data_scale, centers = 2, nstart = 100)
# Assign the cluster labels to km.cluster.
km.cluster <-model$cluster
#Add the result of cluster into dataset
diabetes_dataset_cluster <- data.frame(diabetes_data,cluster = as.factor(model$cluster))
head(diabetes_dataset_cluster)

#Set row names of sample_data_scale by concatenating diabetes category and row numbers from diabetes_data.
rownames(sample_data_scale) <- paste(diabetes_data$diabetes, 1:dim(diabetes_data)[1], sep = '_')
# Visualize the clustering result
fviz_cluster(list(data=sample_data_scale, cluster = km.cluster))
# Create a contingency table to compare the clustering result with the actual diabetes categories.
table(km.cluster, diabetes_data$diabetes)

model <- kmeans(sample_data_scale, centers = 2, nstart = 100) #model with scaled dataset
model2 <- kmeans(sample_data, centers = 2, nstart = 100)# model with unscaled dataset
# measuring the quality BSS/TSS ratio 
# Model with scaled dataset 
(BSS <- model$betweenss)
(TSS <- model$totss)
BSS / TSS * 100
#Model with unscaled dataset
(BSS <- model2$betweenss)
(TSS <- model2$totss)
BSS / TSS * 100

# Replace Cluster 1 (non-diabetes group) with '0' and Cluster 2 (diabetes) with '1' in km.cluster
km.cluster <- ifelse(km.cluster == 1, '0', '1')

#install.packages("caret")
library(caret)
# Convert the predicted labels and actual labels to factors with the same levels
predicted_labels <- factor(km.cluster, levels = c(0, 1))
actual_labels <- factor(diabetes_data$diabetes, levels = c(0, 1))
# Calculate the confusion matrix
confusion_matrix <- confusionMatrix(predicted_labels, actual_labels)

# Print the confusion matrix
print(confusion_matrix)
# Extract the performance metrics
accuracy <- confusion_matrix$overall['Accuracy']
precision <- confusion_matrix$byClass['Precision']
sensitivity <- confusion_matrix$byClass['Sensitivity']
recall <- confusion_matrix$byClass['Recall']
specificity <- confusion_matrix$byClass['Specificity']

# Print the confusion matrix and performance metrics
print(paste("Accuracy:", accuracy))
print(paste("Precision:", precision))
print(paste("Sensitivity (Recall):", sensitivity))
print(paste("Specificity:", specificity))
