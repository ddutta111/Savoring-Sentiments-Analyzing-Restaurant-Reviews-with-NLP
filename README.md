# Savoring-Sentiments-Analyzing-Restaurant-Reviews-with-NLP

## Model Overview:

In this project, we developed a Random Forest classification model for Natural Language Processing (NLP) to analyze restaurant performance based on customer reviews. By leveraging customer feedback, our model aims to accurately predict whether a review is positive or negative, providing valuable insights into customer sentiments. The model employs various text preprocessing techniques to clean and prepare the data, followed by feature extraction using a Bag of Words approach. With a robust evaluation strategy, including confusion matrices and ROC-AUC analysis, we ensure a comprehensive assessment of the model's performance. This approach enables restaurant owners to better understand customer perceptions and enhance their service offerings.

## Dataset Description

The dataset used in this project is the restaurant_reviews.tsv file, sourced from the Maven Analytics available datasets. This collection contains a variety of customer reviews for different restaurants, encompassing a range of sentiments. Each entry in the dataset includes text reviews along with a binary label indicating whether the review is positive or negative. This structured data serves as the foundation for our Natural Language Processing (NLP) model, enabling us to analyze customer opinions and assess restaurant performance effectively.

Statistical software used is R.

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

## Spillting the Dataset into training and test set
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
## Random Forest Model Feature Importance and Visualizing RF result
```R
# Step 12: Random Forest Model Feature Importance
importance <- importance(classifier)
varImpPlot(classifier, main = 'Feature Importance Plot')

# Step 13: Visualizing Random Forest Results
# Plotting the out-of-bag error rate to see if more trees help reduce error
plot(classifier, main = "Error Rate vs. Number of Trees")
```

