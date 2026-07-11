# Iris Flower Classification

## Project Overview

This project is completed as part of my **Data Science Internship at Oasis Infobyte**.

The objective of this task is to build a machine learning classification model that can identify the species of an Iris flower based on its physical measurements. The model classifies the flower into one of the following species:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

The prediction is based on four input features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

## Dataset

The dataset used for this project is the Iris dataset. It contains 150 records and 6 columns.

### Dataset Columns

* Id
* SepalLengthCm
* SepalWidthCm
* PetalLengthCm
* PetalWidthCm
* Species

The target variable is `Species`, which represents the class of the Iris flower.

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Project Workflow

The project was completed using the following steps:

1. Imported the required Python libraries
2. Loaded the Iris dataset
3. Performed Exploratory Data Analysis
4. Checked dataset shape, data types, missing values, and descriptive statistics
5. Checked the distribution of Iris flower species
6. Removed the unnecessary `Id` column
7. Created visualizations using countplot, pairplot, box plots, and correlation heatmap
8. Discussed the most important features for classification
9. Prepared feature variables and target variable
10. Encoded the target variable using LabelEncoder
11. Split the dataset into training and testing sets using an 80/20 split
12. Trained multiple machine learning classification models
13. Evaluated each model using accuracy score, classification report, and confusion matrix
14. Compared the performance of all models
15. Selected the best-performing model
16. Tested the final model using custom input values

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure and quality of the dataset.

The following checks were completed:

* Dataset shape
* Dataset information
* Data types
* Missing values
* Descriptive statistics
* Species count

The dataset was balanced, with 50 records for each Iris species.

## Data Visualization

The following visualizations were created:

* Countplot to show the distribution of flower species
* Pairplot to understand relationships between features
* Box plots to compare feature distribution across different species
* Correlation heatmap to understand relationships between numerical features

From the visualizations, it was observed that `PetalLengthCm` and `PetalWidthCm` are the most useful features for classifying Iris species.

## Feature Selection Discussion

All four numerical features were used for model training:

* SepalLengthCm
* SepalWidthCm
* PetalLengthCm
* PetalWidthCm

Among these features, petal length and petal width showed clearer separation between the three Iris species. Iris-setosa was clearly separated from the other species using petal measurements. Iris-versicolor and Iris-virginica had some overlap, but they could still be classified effectively.

Sepal length and sepal width were also useful, but they showed more overlap compared to petal features. Therefore, all four features were used for training, while petal length and petal width were considered the most discriminative features.

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

The classification report included precision, recall, F1-score, and support for each Iris species.

## Model Performance

| Model                    | Accuracy |
| ------------------------ | -------: |
| Logistic Regression      |      1.0 |
| K-Nearest Neighbors      |      1.0 |
| Decision Tree Classifier |      1.0 |
| Random Forest Classifier |      1.0 |

All four models achieved an accuracy of 1.0 on the test dataset.

## Best Model Selection

All four models performed very well on the test dataset. Logistic Regression, K-Nearest Neighbors, Decision Tree Classifier, and Random Forest Classifier all achieved an accuracy of 1.0.

Since Logistic Regression achieved the best accuracy and is simple, efficient, and easy to interpret, it was selected as the final model for this project. The Iris dataset is clean and has clearly separable feature patterns, especially in petal length and petal width, which helped Logistic Regression classify the species correctly.

## Final Prediction

The final model was tested using custom flower measurements:

* Sepal Length: 5.1
* Sepal Width: 3.5
* Petal Length: 1.4
* Petal Width: 0.2

The model predicted the flower species as:

```text
Iris-setosa
```

## Result

The project successfully trained and evaluated multiple machine learning models for Iris flower classification. All models classified the test samples correctly, and logistic regression was selected as the final model.

## Conclusion

This project helped me understand the complete workflow of a machine learning classification problem. I learned how to load a dataset, perform exploratory data analysis, create visualizations, identify important features, train multiple classification models, evaluate model performance, compare models, and make predictions using custom input values.

## Internship

This project was completed as part of the **Oasis Infobyte Data Science Internship**.

## Author

**Bhanu Prakash Nambari**
