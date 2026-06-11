# Task 5: Sales Prediction Using Python

## Project Overview

This project is part of my **Data Science Internship at Oasis Infobyte**.

The main objective of this task is to build a machine learning model that can predict product sales based on advertising budget. Businesses spend money on different advertising platforms such as TV, Radio, and Newspaper. This project helps understand how these advertising channels affect sales and how machine learning can be used for sales prediction.

In this project, I used Python and Linear Regression to predict sales using advertising data.

---

## Project Title

**Sales Prediction Using Python**

---

## Objective

The objective of this project is to predict sales based on the amount spent on different advertising channels.

The input features used in this project are:

* TV advertising budget
* Radio advertising budget
* Newspaper advertising budget

The target variable is:

* Sales

---

## Dataset Description

The dataset used in this project is named:

```text
Advertising.csv
```

The dataset contains advertising budget details and sales values.

### Dataset Columns

| Column Name | Description                                |
| ----------- | ------------------------------------------ |
| Unnamed: 0  | Index column, removed during data cleaning |
| TV          | Advertising budget spent on TV             |
| Radio       | Advertising budget spent on Radio          |
| Newspaper   | Advertising budget spent on Newspaper      |
| Sales       | Sales generated                            |

The column `Unnamed: 0` was removed because it was only an index column and not useful for prediction.

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

### Linear Regression

Linear Regression was used in this project because the target variable, Sales, is a continuous numerical value.

The model learns the relationship between advertising budgets and sales, then predicts sales for new advertising budget values.

---

## Project Workflow

The project was completed using the following steps:

1. Created the project folder
2. Added the dataset file `Advertising.csv`
3. Created the Jupyter Notebook file `sales_prediction.ipynb`
4. Imported the required Python libraries
5. Loaded the dataset
6. Checked dataset shape, information, and missing values
7. Removed the unnecessary `Unnamed: 0` column
8. Performed basic statistical analysis
9. Visualized the relationship between advertising channels and sales
10. Checked correlation between variables
11. Selected input features and target variable
12. Split the dataset into training and testing sets
13. Trained a Linear Regression model
14. Predicted sales using test data
15. Compared actual sales and predicted sales
16. Evaluated the model using MAE, MSE, RMSE, and R2 Score
17. Tested the model with new advertising budget values

---

## Data Cleaning

The dataset contained an unnecessary column named:

```text
Unnamed: 0
```

This column was removed because it was only an index column and did not contribute to sales prediction.

After cleaning, the final columns used were:

```text
TV
Radio
Newspaper
Sales
```

---

## Data Visualization

Scatter plots were created to understand the relationship between advertising budget and sales.

The following relationships were visualized:

* TV Advertising vs Sales
* Radio Advertising vs Sales
* Newspaper Advertising vs Sales
* Actual Sales vs Predicted Sales

These visualizations helped to understand which advertising channels had a stronger relationship with sales.

---

## Feature Selection

The independent variables used for prediction were:

```text
TV
Radio
Newspaper
```

The dependent variable was:

```text
Sales
```

---

## Model Training

The dataset was split into training and testing data.

* Training data was used to train the model
* Testing data was used to evaluate the model performance

A Linear Regression model was trained using the training data.

---

## Model Evaluation

The model was evaluated using the following metrics:

| Metric                  | Value              |
| ----------------------- | ------------------ |
| Mean Absolute Error     | 1.4607567168117606 |
| Mean Squared Error      | 3.1740973539761046 |
| Root Mean Squared Error | 1.7815996615334502 |
| R2 Score                | 0.899438024100912  |

---

## Result

The model achieved an **R2 Score of 0.8994**, which means the model explains around **89.94%** of the variation in sales.

This shows that the model performed well in predicting sales based on advertising budget.

---

## Sample Prediction

A sample prediction was made using new advertising budget values.

Example input:

| TV  | Radio | Newspaper |
| --- | ----- | --------- |
| 150 | 30    | 20        |

The trained model predicts the expected sales value based on these advertising budgets.

---

## Conclusion

In this project, I successfully built a Sales Prediction model using Python and Linear Regression. The dataset was cleaned by removing the unnecessary index column, and the relationship between advertising channels and sales was analyzed using visualizations.

The model was trained using TV, Radio, and Newspaper advertising budgets as input features. The model achieved an R2 Score of 0.8994, which shows good prediction performance.

This project helped me understand how machine learning can be used in business decision-making, especially for predicting sales based on marketing investment.

---

## Files in This Repository

```text
Task-5-Sales-Prediction/
│
├── Advertising.csv
├── sales_prediction.ipynb
└── README.md
```

---

## How to Run This Project

1. Download or clone this repository
2. Open the `Task-5-Sales-Prediction` folder
3. Make sure `Advertising.csv` is available in the same folder
4. Open `sales_prediction.ipynb` using Jupyter Notebook
5. Run all cells step by step
6. Check the model evaluation results and sales prediction output

## Author:

Bhanu Prakash Nambari
* Programming Language: Python

