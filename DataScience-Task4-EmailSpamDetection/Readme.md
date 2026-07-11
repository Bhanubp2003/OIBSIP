# Email Spam Detection with Machine Learning

## Project Overview

This project is part of my Oasis Infobyte Data Science Internship. The objective of this project is to build a Natural Language Processing (NLP) binary classification model that can classify messages as spam or ham.

Spam detection is an important machine learning application because it helps identify unwanted, promotional, scam, or harmful messages before they reach the user's inbox.

## Objective

To build a machine learning model that can distinguish spam emails/messages from legitimate ham emails/messages using Natural Language Processing techniques.

## Tech Stack Used

- Python
- pandas
- NumPy
- scikit-learn
- re
- matplotlib
- seaborn
- WordCloud
- Jupyter Notebook

## Dataset

The dataset used in this project contains labeled text messages categorized as:

- ham: legitimate message
- spam: unwanted or promotional message

The dataset was loaded using pandas and cleaned before model training.

## Project Workflow

1. Loaded the spam detection dataset
2. Checked dataset shape, missing values, and duplicate records
3. Checked spam vs ham class distribution using counts and percentages
4. Applied text preprocessing:
   - Lowercase conversion
   - Punctuation removal
   - Number removal
   - Extra space removal
   - Stopword removal
5. Converted text data into numerical features using TF-IDF Vectorizer
6. Split the data into training and testing sets
7. Trained multiple machine learning models:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Support Vector Machine
8. Evaluated models using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
9. Compared model performance
10. Tested the model using custom sample messages

## TF-IDF Explanation

TF-IDF stands for Term Frequency-Inverse Document Frequency. It measures how important a word is in a message compared to the entire dataset.

Term Frequency measures how often a word appears in a message, while Inverse Document Frequency reduces the importance of words that appear too commonly across many messages.

TF-IDF is useful in spam detection because spam messages often contain important repeated words such as free, win, cash, prize, claim, and urgent.

## Models Used

### Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification problems. It performs well with word frequency-based features such as TF-IDF.

### Logistic Regression

Logistic Regression is a strong baseline algorithm for binary classification tasks and was used as an alternative classifier.

### Support Vector Machine

Support Vector Machine was also trained as an additional classifier to improve model comparison.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Why Recall Is Important in Spam Detection

Recall is especially important in spam detection because it measures how many actual spam messages are correctly identified by the model.

If recall is low, many spam messages may be incorrectly classified as ham. This can be risky because harmful emails such as phishing links, fake offers, scam messages, or malware-related content may reach the user's inbox.

Therefore, a good spam detection model should try to achieve high recall while also maintaining reasonable precision.

## Results

The trained models successfully classified messages as spam or ham. The evaluation metrics showed that machine learning models can effectively detect spam messages using NLP and TF-IDF-based text features.

## Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be used for spam detection. Through this task, I learned how to preprocess text data, extract features using TF-IDF, train multiple classifiers, evaluate model performance, and compare results using different metrics.

## Internship

This project was completed as part of the Oasis Infobyte Data Science Internship.

## Author

Bhanu Prakash Nambari
