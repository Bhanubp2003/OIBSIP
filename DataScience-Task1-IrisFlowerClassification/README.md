# Iris Flower Classification

## Project Overview

This project is part of my Data Science Internship at Oasis Infobyte. The objective of this task is to build a machine learning model that can classify Iris flowers into three species: Iris-setosa, Iris-versicolor, and Iris-virginica.

The classification is based on four flower measurements: sepal length, sepal width, petal length, and petal width.

## Dataset

The dataset used in this project is the Iris dataset. It contains 150 records with the following columns:

- Id
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species

The target column is `Species`, which represents the flower class.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Model

The model used for this project is:

- Random Forest Classifier

Random Forest was selected because it performs well for classification tasks and is suitable for structured datasets like the Iris dataset.

## Project Workflow

1. Imported required Python libraries
2. Loaded the Iris dataset
3. Explored the dataset using shape, info, and statistical summary
4. Checked for missing values
5. Visualized the dataset using countplot, pairplot, and correlation heatmap
6. Removed unnecessary columns
7. Prepared feature variables and target variable
8. Encoded the target variable
9. Split the dataset into training and testing data
10. Trained the Random Forest Classifier model
11. Made predictions on test data
12. Evaluated the model using accuracy, classification report, and confusion matrix
13. Tested the model using custom flower measurements

## Result

The model achieved an accuracy of 100% on the test dataset. The confusion matrix also showed that all test samples were classified correctly without any misclassification.

## Conclusion

This project helped me understand the basic workflow of a machine learning classification problem. I learned how to load data, explore it, visualize patterns, train a classification model, evaluate model performance, and make predictions using new input values.

## Internship

This project was completed as part of the Oasis Infobyte Data Science Internship.

## Author

Bhanu Prakash Nambari