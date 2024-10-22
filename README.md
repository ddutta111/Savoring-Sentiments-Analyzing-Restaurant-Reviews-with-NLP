# Savoring-Sentiments-Analyzing-Restaurant-Reviews-NLP-R

## Model Overview:

In this project, we developed a Random Forest classification model for Natural Language Processing (NLP) to analyze restaurant performance based on customer reviews. By leveraging customer feedback, our model aims to accurately predict whether a review is positive or negative, providing valuable insights into customer sentiments. The model employs various text preprocessing techniques to clean and prepare the data, followed by feature extraction using a Bag of Words approach. With a robust evaluation strategy, including confusion matrices and ROC-AUC analysis, we ensure a comprehensive assessment of the model's performance. This approach enables restaurant owners to better understand customer perceptions and enhance their service offerings.

## Dataset Description

The dataset used in this project is the restaurant_reviews.tsv file, sourced from the Maven Analytics available datasets. This collection contains a variety of customer reviews for different restaurants, encompassing a range of sentiments. Each entry in the dataset includes text reviews along with a binary label indicating whether the review is positive or negative. This structured data serves as the foundation for our Natural Language Processing (NLP) model, enabling us to analyze customer opinions and assess restaurant performance effectively.

Statistical software R is for data analysis, generate insight and by building the ML model.

## Imorting the dataset
```R
# Step 1: Importing the dataset
dataset_original = read.delim('D:\\R Projects\\Restaurant_Reviews.tsv', quote = '', 
                              stringsAsFactors = FALSE)
```
## Exploratory Data Analysis
```R
#Step 2: EDA
#Check the dataset structure
str(dataset_original)
```
The dataset dataset_original is a data frame with 1000 observations (rows) and 2 variables (columns):

Review (chr): This column contains textual reviews of restaurants. Each review is stored as a character string (chr), where customers express their thoughts about the restaurant (e.g., "Wow... Loved this place.", "Crust is not good.").

Liked (int): This column is a binary indicator representing whether the customer liked the restaurant or not. It is an integer value:

1 means the customer liked the restaurant (positive review).
0 means the customer did not like the restaurant (negative review).

In summary, the dataset captures customer sentiments toward restaurants through textual reviews and corresponding binary labels that indicate whether the review was positive or negative.

## Data Pre-Processing

```R
# Checking the distribution of the target variable ('Liked')
table(dataset_original$Liked)
```
Result:

500 reviews have a Liked value of 0, meaning these are negative reviews.
500 reviews have a Liked value of 1, meaning these are positive reviews.
This result indicates a perfectly balanced dataset, with an equal number of positive and negative reviews (500 each). This balance is beneficial for training classification models, as it avoids bias toward one class.

```R
# Plotting the distribution of 'Liked' variable
library(ggplot2)
ggplot(dataset_original, aes(x = as.factor(Liked))) +
  geom_bar(fill = 'blue') +
  labs(title = 'Distribution of Reviews', x = 'Liked', y = 'Count') +
  theme_minimal()
```
![Distribution of Reviews](https://github.com/user-attachments/assets/afaa5e39-2322-46f9-9ba1-259ec363a2d3)

This graph is a bar plot showing the distribution of restaurant reviews categorized by whether customers liked the restaurant (labeled as 1) or did not like it (labeled as 0).

- The x-axis represents whether the review indicates that the customer liked (1) or did not like (0) the restaurant.
- The y-axis shows the count of reviews for each category.

Both bars have roughly the same height, indicating that there is a nearly equal number of positive (liked) and negative (not liked) reviews, with approximately 500 reviews in each category. This balanced distribution is useful for training machine learning models, as it avoids bias toward one class.

Data Checking

```R
# Checking word frequency before cleaning
install.packages('wordcloud')
library(wordcloud)
corpus_raw <- VCorpus(VectorSource(dataset_original$Review))
dtm_raw <- DocumentTermMatrix(corpus_raw)
word_freq <- sort(colSums(as.matrix(dtm_raw)), decreasing = TRUE)

# Plotting a word cloud for the most frequent words (before cleaning)
set.seed(123)
wordcloud(names(word_freq), freq = word_freq, max.words = 100, colors = brewer.pal(8, 'Dark2'))
```
![Word Cloud](https://github.com/user-attachments/assets/6535bb2f-1b99-4fea-8bc7-7da420209cdb)

The "word cloud" visualizes the frequency of words in restaurant reviews, with larger words appearing more often. Common words like "the", "was", and "and" are not insightful. The words "food", "place", "service", and "good" highlight common themes in restaurant reviews, focusing on food and service quality. Terms like "not", "very", and "love" suggest strong positive or negative opinions. Words such as "friendly", "staff", and "minutes" indicate customer service and wait time feedback. 

This type of word cloud can provide a high-level overview of the most talked-about aspects of a restaurant, though further NLP techniques such as sentiment analysis would be needed for deeper insights.

Data Cleaning
```R
# Step 3: Text Preprocessing and Cleaning
# Loading required libraries for text cleaning
library(tm)          # For text mining
library(SnowballC)   # For stemming (reducing words to their root forms)

# Creating a corpus (collection of text documents)
corpus <- VCorpus(VectorSource(dataset_original$Review))

# Converting text to lowercase
corpus <- tm_map(corpus, content_transformer(tolower))

# Removing numbers
corpus <- tm_map(corpus, removeNumbers)

# Removing punctuation marks
corpus <- tm_map(corpus, removePunctuation)

# Removing common stop words like 'the', 'and', etc.
corpus <- tm_map(corpus, removeWords, stopwords())

# Applying stemming to get root words (e.g., "loved" becomes "love")
corpus <- tm_map(corpus, stemDocument)
```
## Bag Of Words Model Creation 
```R
# Step 4: Creating the Bag of Words model
dtm <- DocumentTermMatrix(corpus)  # Creating a document-term matrix (word frequency table)
dtm <- removeSparseTerms(dtm, 0.999)  # Removing sparse terms (terms that appear in very few documents)

# Converting the DTM to a dataframe
dataset <- as.data.frame(as.matrix(dtm))

# Adding the 'Liked' column from the original dataset to the cleaned data
dataset$Liked <- dataset_original$Liked

# Removing extra white spaces
corpus <- tm_map(corpus, stripWhitespace)
```
## Encoding Categorical Feature
```R
# Step 5: Encoding the target feature as a factor
dataset$Liked <- factor(dataset$Liked, levels = c(0, 1))
# Explanation: The 'Liked' column is encoded as a factor with two levels: 0 (negative review) and 1 (positive review).
```

## Splitting the Dataset into training and test set
```R
#Step 6: Splitting Dataset for model validation 
library(caTools)
set.seed(123)  # Setting a random seed for reproducibility
split <- sample.split(dataset$Liked, SplitRatio = 0.8)  # 80% Training, 20% Test

training_set <- subset(dataset, split == TRUE)  # Training set
test_set <- subset(dataset, split == FALSE)     # Test set
```
## Randomforest Classification Model
```R
# Step 7: Fitting Random Forest Classification to the Training set
library(randomForest)
classifier <- randomForest(x = training_set[-692],  # All columns except the target variable
                           y = training_set$Liked,  # Target variable
                           ntree = 10)  # Number of trees in the forest
# Explanation: Random Forest is used as the classification algorithm with 10 trees.

# Summary of the Random Forest model
print(classifier)
```
Random Forest model's performance on the training dataset result: 
```R
> print(classifier)

Call:
 randomForest(x = training_set[-692], y = training_set$Liked,      ntree = 10) 
               Type of random forest: classification
                     Number of trees: 10
No. of variables tried at each split: 26

        OOB estimate of  error rate: 27.9%
Confusion matrix:
    0   1 class.error
0 304  90   0.2284264
1 131 267   0.3291457
```
Result Explanation:

- The model was trained with 10 trees.
- At each split, 26 variables were considered to determine the best split.
- Out-Of-Bag (OOB) Error Rate:

- The OOB error rate is 27.9%, which indicates that approximately 27.9% of the training samples were misclassified when they were left out of the bootstrap samples during training. This is a good estimate of the model's error on unseen data.

Confusion Matrix on Training Set:

- True Negatives (TN): 304 reviews were correctly classified as negative (class 0).
- False Positives (FP): 90 reviews were incorrectly classified as positive when they were actually negative.
- False Negatives (FN): 131 reviews were incorrectly classified as negative when they were actually positive.
- True Positives (TP): 267 reviews were correctly classified as positive (class 1).

Class Error Rates:

- Class 0 (Negative Reviews): The error rate is 22.84%, meaning 90 out of 394 negative reviews were misclassified.
- Class 1 (Positive Reviews): The error rate is 32.91%, meaning 131 out of 398 positive reviews were misclassified.

- Conclusion
  
The Random Forest classifier showed a moderate performance on the training dataset with a relatively higher error rate for positive reviews (class 1) compared to negative reviews (class 0). The OOB error rate of 27.9% provides a reasonable estimate of how the model would perform on unseen data.

## Prediction on Test Set
```R
# Step 8: Predicting the Test set results
y_pred <- predict(classifier, newdata = test_set[-692])
print(y_pred)
# Explanation: The trained Random Forest model is used to predict the 'Liked' label for the test set.
```
## Confusion Matrix & Model Evaluation
```R
# Step 9: Making the Confusion Matrix
cm <- table(test_set[, 692], y_pred)
print(cm)
# Explanation: A confusion matrix is generated to compare the actual results with the predicted results.

# Step 10: Model Evaluation - Accuracy, Precision, Recall, F1-score
library(caret)
confusionMatrix(y_pred, test_set$Liked)
```
Confusion Matrix Result Explanation (Test Dataset):

The confusion matrix evaluates the Random Forest classifier's performance on the test dataset, where the predicted labels (y_pred) are compared to the actual labels from the test set. The output is below:
```R
> print(cm)
   y_pred
     0  1
  0 82 18
  1 23 77
```
Explnation of the Confusion Matrix result on test dataset is:

Actual / Predicted	   Predicted 0 (Disliked)	           Predicted 1 (Liked)
Actual 0 (Disliked)    	 82 (True Negatives)	          18 (False Positives)
Actual 1 (Liked)	      23 (False Negatives)	           77 (True Positives)

- True Negatives (TN) = 82: The model correctly predicted 82 reviews as negative (disliked), and these reviews were actually negative.
- False Positives (FP) = 18: The model incorrectly predicted 18 reviews as positive (liked), but these reviews were actually negative.
- False Negatives (FN) = 23: The model incorrectly predicted 23 reviews as negative (disliked), but these reviews were actually positive.
- True Positives (TP) = 77: The model correctly predicted 77 reviews as positive (liked), and these reviews were actually positive.

Model Evaluation - Accuracy, Precision, Recall, F1-scores are:

- Accuracy rate (overall correctness of the model) is 79.5%.
- Precision rate (for class 1, liked reviews - measures how many predicted positive instances were actually positive) is 81%.
- Recall score (for class 1, liked reviews - measures how many actual positive instances were correctly predicted) is 77%.
- F1-Score (A balance between precision and recall) is 79%.
  
## ROC-AUC Evaluation
```R
Step 11: # Install and load the pROC package for AUC
# install.packages('pROC')
library(pROC)

# Step 11.1: Predicting probabilities instead of classes
y_prob <- predict(classifier, newdata = test_set[-ncol(test_set)], type = "prob")[, 2]

# Step 11.2: Computing the ROC curve
roc_curve <- roc(test_set$Liked, y_prob)

# Step 11.3: Plotting the ROC curve
plot(roc_curve, col = "blue", main = "ROC Curve for Random Forest")

# Step 11.4: Displaying the AUC (Area Under the Curve)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))
# Displaying the AUC
auc(roc_curve)
```
ROC-AUC Result explanation:

- The ROC-AUC analysis evaluates the Random Forest model's performance in predicting probabilities rather than direct classifications.

**ROC Curve:** A graphical representation of the model's ability to distinguish between positive and negative classes (liked vs. disliked reviews). The curve plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings.

**AUC (Area Under the Curve) = 0.8444:** This value indicates that the model has an 84.44% chance of correctly distinguishing between positive and negative reviews. A higher AUC value (close to 1) indicates better model performance, while a value around 0.5 would mean random guessing.

In this case, the **AUC of 0.8444 suggests that the Random Forest model performs well in distinguishing between liked and disliked reviews.**

![ROC Curve](https://github.com/user-attachments/assets/e5ef68df-9234-4580-87f0-371b84e567ec)

- The ROC curve (Receiver Operating Characteristic curve) - plot illustrates the trade-off between sensitivity (true positive rate) and specificity (true negative rate) for a binary classification model. In the context of an NLP model for random forest, the ROC curve shows how well the model can distinguish between positive and negative instances (e.g., positive and negative reviews).

> The key components of the ROC curve:

- Sensitivity: The proportion of positive instances that are correctly classified as positive.
- Specificity: The proportion of negative instances that are correctly classified as negative.
- Diagonal line: This represents a classifier that randomly guesses the class of each instance.
- Curve: The shape of the curve indicates the performance of the model. A curve that is closer to the top-left corner indicates better performance, as it means the model has high sensitivity and specificity. A curve that is closer to the diagonal line indicates poor performance. Here, teh curve shows that our model performs moderately well.

## Random Forest Model Feature Importance and Visualizing RF result
```R
# Step 12: Random Forest Model Feature Importance
importance <- importance(classifier)
varImpPlot(classifier, main = 'Feature Importance Plot')
```
The Feature Importance Plot shows the relative importance of different words (features) for a classification model, likely built for predicting whether a restaurant review is positive or negative.

![Feature Importance Plot for RF model](https://github.com/user-attachments/assets/271f5add-ea67-429a-9616-aefff552dcb1)

> Key takeaways:

- Words at the top of the plot, like "great", "good", and "delici" (delicious), have the highest feature importance. This means they are more influential in determining whether a review is positive.
- Words lower on the plot, like "fun", "breakfast", and "realli" (really), have less importance in the model.
- Positive adjectives such as "fantast" (fantastic), "awesom" (awesome), and "best" are crucial indicators of positive reviews.
- Negative words like "bad", "worst", and "overpr" (overpriced) play a role in identifying negative reviews.

This plot helps explain which words the model relies on most for making predictions.

```R
# Step 13: Visualizing Random Forest Results
# Plotting the out-of-bag error rate to see if more trees help reduce error
plot(classifier, main = "Error Rate vs. Number of Trees")
```

![Error Rate vs Trees](https://github.com/user-attachments/assets/32899c35-487c-4e09-937d-d6baad4f740c)

Here, the plot shows that the error rate generally decreases as the number of trees increases. This is because more trees can help the model to learn more complex patterns in the data. However, there is a point of diminishing returns, after which adding more trees does not significantly improve the error rate.The plot also shows that the error rate varies across different random forest models. This is because random forest models are stochastic, meaning that they are sensitive to the random initialization of the trees. As a result, different random forest models can have slightly different error rates.

## Conclusion

The Random Forest classification model effectively analyzed restaurant reviews, achieving an overall accuracy of 79.5% on the test dataset. The balanced distribution of positive and negative reviews contributed to the model's robust performance, evidenced by a precision rate of 81% and a recall rate of 77% for liked reviews. While the model demonstrated strong capabilities in predicting negative reviews with a lower error rate (22.84%), it struggled more with positive reviews, reflecting a higher error rate of 32.91%.

The Out-Of-Bag (OOB) error rate of 27.9% indicates a reasonable level of generalization for unseen data. Additionally, the ROC-AUC score of 0.8444 suggests that the model has a good ability to distinguish between liked and disliked reviews.

Visualizations, such as the error rate versus the number of trees, illustrate that increasing the number of trees generally reduces error rates, underscoring the model's ability to capture complex patterns in the data. Overall, this analysis highlights the utility of the Random Forest classifier in sentiment analysis and provides a foundation for further exploration of customer opinions in the restaurant industry through advanced NLP techniques.

Overall, our Random Forest Classifier model perform moderately good. However, the below strategies can enhance the model's accuracy and performance further such as -
Hyperparameter Tuning: Optimize parameters like the number of trees and depth; Class Imbalance Handling: Use SMOTE or adjust class weights to balance classes; Advanced Text Processing: Apply TF-IDF or word embeddings for better representation; Cross-Validation: Implement k-fold cross-validation for robust performance estimates; NLP Techniques: using LSTM or Transformer models for deeper context understanding, etc.

These strategies can enhance the model's accuracy and generalization.

## Recommendations for the Restaurant Owner

Based on the restaurant review analysis we can have an insight of customer sentiment about the restaurant's overall performance, hence following recommendations are suggested to the client for improvement in it's profit:

1. Improve Customer Experience: Enhance food quality and service speed through staff training.
2. Gather Feedback: Implement post-meal surveys to identify areas for improvement.
3. Analyze Negative Reviews: Address recurring issues highlighted in customer feedback.
4. Leverage Positive Feedback: Use positive reviews in marketing efforts to attract new customers.
5. Engage Online: Respond to customer reviews on social media and review platforms.
6. Adjust Menu: Focus on popular dishes and consider phasing out less favored items.
7. Regular Staff Training: Ensure consistent service quality through ongoing staff training.
8. Run Promotions: Create targeted promotional campaigns to attract different customer segments.

Author: Debolina Dutta

LinkedIn: https://www.linkedin.com/in/duttadebolina/
