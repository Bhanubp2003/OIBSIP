# Task 4: Email Spam Detection Using Machine Learning

## Project Overview

This project is part of my **Data Science Internship at Oasis Infobyte**.

The main objective of this task is to build a machine learning model that can detect whether a given message is **Spam** or **Ham**. Spam messages usually include unwanted advertisements, fake offers, fraud messages, or promotional content, while Ham messages are normal and useful messages.

In this project, I used Python and Natural Language Processing techniques to classify messages based on their text content.

---

## Project Title

**Email Spam Detection Using Machine Learning**

---

## Objective

The objective of this project is to create a machine learning model that can automatically classify text messages as either spam or ham.

This type of model is useful in real-world applications such as:

* Email filtering
* SMS spam detection
* Fraud message identification
* Message classification systems

---

## Dataset Description

The dataset used in this project is named:

```text
spam.csv
```

The dataset contains SMS messages with labels showing whether each message is spam or ham.

### Dataset Columns

| Column Name | Description                              |
| ----------- | ---------------------------------------- |
| v1          | Label of the message, either ham or spam |
| v2          | Actual text message                      |
| Unnamed: 2  | Unnecessary column                       |
| Unnamed: 3  | Unnecessary column                       |
| Unnamed: 4  | Unnecessary column                       |

Only the important columns `v1` and `v2` were used for this project. The remaining unnecessary columns were removed during data cleaning.

---

## Technologies and Libraries Used

The following technologies and Python libraries were used in this project:

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Machine Learning Algorithm Used

For this project, I used the following machine learning algorithm:

### Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for classification problems. Since this project classifies messages into two categories, Spam or Ham, Logistic Regression is suitable for this task.

---

## Project Workflow

The project was completed using the following steps:

1. Imported the required Python libraries
2. Loaded the `spam.csv` dataset
3. Checked the shape and basic information of the dataset
4. Removed unnecessary columns from the dataset
5. Renamed the useful columns for better understanding
6. Checked for missing values
7. Explored the count of spam and ham messages
8. Converted text labels into numerical values
9. Split the data into training and testing sets
10. Converted text messages into numerical features using TF-IDF Vectorizer
11. Trained the Logistic Regression model
12. Made predictions using the test data
13. Evaluated the model using accuracy score, classification report, and confusion matrix
14. Tested the model with custom messages

---

## Data Cleaning

The original dataset contained extra columns that were not useful for prediction. So, only the required columns were selected:

```text
v1 - Label
v2 - Message
```

These columns were renamed as:

```text
label
message
```

The labels were converted into numerical values:

| Label | Numerical Value |
| ----- | --------------- |
| Ham   | 0               |
| Spam  | 1               |

---

## Text Preprocessing and Feature Extraction

Machine learning models cannot understand text directly. So, the text messages were converted into numerical form using **TF-IDF Vectorizer**.

### TF-IDF Vectorizer

TF-IDF stands for **Term Frequency-Inverse Document Frequency**. It helps convert text data into numerical values by giving importance to words based on how frequently they appear in a message and across the dataset.

This step helped the machine learning model understand the message content and classify it correctly.

---

## Model Training

The dataset was divided into training and testing data.

* Training data was used to train the model
* Testing data was used to check how well the model performs on unseen messages

The Logistic Regression model was trained using the TF-IDF features of the messages.

---

## Model Evaluation

The model performance was evaluated using the following metrics:

* Accuracy Score
* Classification Report
* Confusion Matrix

These metrics helped to understand how correctly the model classified spam and ham messages.

---

## Sample Prediction

### Example 1

Input message:

```text
Congratulations! You have won a free iPhone. Click here to claim now.
```

Output:

```text
This message is Spam
```

### Example 2

Input message:

```text
Hey, are we meeting tomorrow for lunch?
```

Output:

```text
This message is Ham
```

---

## Results

The model was able to classify messages as spam or ham successfully. The accuracy score showed that the model performed well on the test data.

The use of TF-IDF Vectorizer helped convert text into meaningful numerical features, and Logistic Regression provided good classification results.

---

## Conclusion

In this project, I successfully built an Email Spam Detection model using machine learning. The dataset was cleaned, unnecessary columns were removed, and the text messages were converted into numerical format using TF-IDF Vectorizer.

A Logistic Regression model was trained to classify messages as Spam or Ham. The model was evaluated using accuracy score, classification report, and confusion matrix.

This project helped me understand how Natural Language Processing and machine learning can be used to solve real-world text classification problems.

---

## Files in This Repository

```text
Task-4-Email-Spam-Detection/
│
├── spam.csv
├── email_spam_detection.ipynb
└── README.md
```

---

## How to Run This Project

1. Download or clone this repository
2. Open the project folder
3. Open the Jupyter Notebook file
4. Make sure the dataset `spam.csv` is in the same folder
5. Run all the cells step by step
6. Check the model accuracy and predictions

---

## Author
Bhanu Prakash Nambari

