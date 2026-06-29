# Iris Flower Classification

## Project Overview

This project is part of my Data Science Internship at Oasis Infobyte. The objective of this task is to build a machine learning classification model that can identify the species of an Iris flower based on its physical measurements.

The model classifies Iris flowers into three species:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

The classification is based on the following flower measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

## Dataset

The dataset used in this project is the Iris dataset. It contains 150 records and includes the following columns:

* Id
* SepalLengthCm
* SepalWidthCm
* PetalLengthCm
* PetalWidthCm
* Species

The target variable is `Species`, which represents the flower class.

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Project Workflow

1. Imported required Python libraries
2. Loaded the Iris dataset
3. Performed Exploratory Data Analysis
4. Checked dataset shape, data types, missing values, and descriptive statistics
5. Visualized the data using countplot, pairplot, box plots, and correlation heatmap
6. Discussed important features for classification
7. Removed unnecessary columns
8. Prepared feature variables and target variable
9. Encoded the target variable using LabelEncoder
10. Split the dataset into training and testing sets
11. Trained multiple machine learning classification models
12. Evaluated each model using accuracy score, classification report, and confusion matrix
13. Compared model performance
14. Selected the best-performing model
15. Tested the model using custom flower measurements

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and quality of the dataset. The dataset was checked for:

* Number of rows and columns
* Data types
* Missing values
* Statistical summary
* Species distribution

The dataset was balanced, with 50 records for each Iris species.

## Data Visualization

The following visualizations were used in this project:

* Countplot to show the distribution of flower species
* Pairplot to understand relationships between features
* Box plots to compare feature distribution across species
* Correlation heatmap to identify relationships between numerical features

From the visualizations, it was observed that `PetalLengthCm` and `PetalWidthCm` are the most useful features for separating the Iris species.

## Feature Selection Discussion

All four numerical features were used for model training:

* SepalLengthCm
* SepalWidthCm
* PetalLengthCm
* PetalWidthCm

However, petal length and petal width were found to be more discriminative compared to sepal length and sepal width. Iris-setosa was clearly separated from the other species using petal measurements, while Iris-versicolor and Iris-virginica showed some overlap but could still be classified effectively.

## Machine Learning Models Used

The following classification models were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree Classifier
* Random Forest Classifier

## Model Evaluation

Each model was evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix

The classification report included precision, recall, and F1-score for each Iris species.

## Results

All four models achieved an accuracy of 1.0 on the test dataset.

| Model                    | Accuracy |
| ------------------------ | -------: |
| Logistic Regression      |      1.0 |
| K-Nearest Neighbors      |      1.0 |
| Decision Tree Classifier |      1.0 |
| Random Forest Classifier |      1.0 |

The confusion matrix showed that the test samples were classified correctly without misclassification.

## Best Model Selection

All models achieved the same accuracy on the test dataset. Since the performance was equal, Random Forest Classifier was selected as the final model because it is an ensemble learning method and generally performs well on classification problems.

Random Forest combines multiple decision trees and helps reduce the risk of overfitting compared to a single decision tree. Therefore, it was considered the final model for this project.

## Final Prediction

The trained model was also tested using custom flower measurements. Based on the given input values, the model successfully predicted the Iris species.

## Conclusion

This project helped me understand the complete workflow of a machine learning classification problem. I learned how to load and explore data, perform visual analysis, identify important features, train multiple classification models, evaluate performance, compare models, and select the best-performing model.

## Internship

This project was completed as part of the Oasis Infobyte Data Science Internship.

## Author

Bhanu Prakash Nambari
