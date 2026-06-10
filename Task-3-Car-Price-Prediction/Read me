# Task 3: Car Price Prediction Using Machine Learning

## Project Overview

This project is part of my **Oasis Infobyte Data Science Internship**.

In this project, I built a machine learning model to predict the **selling price of used cars** based on different features like present price, kilometers driven, fuel type, selling type, transmission type, owner details, and car age.

Car price prediction is a regression problem because the output value is a continuous number.

---

## Objective

The main objective of this project is to predict the selling price of a used car using machine learning.

Through this project, I learned how to:

- Load and understand a dataset
- Clean and preprocess data
- Create a new feature from existing data
- Convert categorical data into numerical data
- Train regression models
- Evaluate model performance
- Compare actual and predicted prices

---

## Dataset Information

The dataset contains details about used cars.

### Columns in the Dataset

| Column Name | Description |
|---|---|
| Car_Name | Name of the car |
| Year | Year of purchase |
| Selling_Price | Price at which the car is being sold |
| Present_Price | Current price of the car |
| Driven_kms | Total kilometers driven |
| Fuel_Type | Type of fuel used |
| Selling_type | Dealer or individual seller |
| Transmission | Manual or automatic transmission |
| Owner | Number of previous owners |

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Models Used

For this project, I used regression models because the target variable is `Selling_Price`.

The models used are:

- Linear Regression
- Decision Tree Regressor

I used these models instead of Random Forest because Random Forest was already used in my previous task.

---

## Steps Followed

### 1. Imported Required Libraries

First, I imported the necessary Python libraries for data analysis, visualization, and machine learning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

### 2. Loaded the Dataset

The car dataset was loaded using Pandas.

```python
df = pd.read_csv("car data.csv")
```

---

### 3. Explored the Dataset

I checked the dataset using basic Pandas functions.

```python
df.head()
```

```python
df.info()
```

```python
df.describe()
```

```python
df.shape
```

This helped me understand the structure of the dataset.

---

### 4. Checked Missing Values

I checked whether the dataset had any missing values.

```python
df.isnull().sum()
```

---

### 5. Checked Duplicate Values

I checked and removed duplicate rows if any were present.

```python
df.duplicated().sum()
```

```python
df = df.drop_duplicates()
```

---

### 6. Created Car Age Column

Instead of using only the year, I created a new column called `Car_Age`.

```python
df["Car_Age"] = 2018 - df["Year"]
```

This helps the model understand how old the car is.

---

### 7. Removed Unnecessary Columns

The `Car_Name` and `Year` columns were removed because they were not needed for the final model.

```python
df = df.drop(["Car_Name", "Year"], axis=1)
```

---

### 8. Encoded Categorical Columns

Text columns like fuel type, selling type, and transmission were converted into numerical format.

```python
df = pd.get_dummies(df, drop_first=True)
```

---

### 9. Selected Features and Target

The target column was `Selling_Price`.

```python
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
```

---

### 10. Split the Dataset

The dataset was split into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 11. Trained the Linear Regression Model

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```

---

### 12. Made Predictions Using Linear Regression

```python
lr_pred = lr_model.predict(X_test)
```

---

### 13. Evaluated Linear Regression Model

```python
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression MAE:", lr_mae)
print("Linear Regression MSE:", lr_mse)
print("Linear Regression RMSE:", lr_rmse)
print("Linear Regression R2 Score:", lr_r2)
```

---

### 14. Trained the Decision Tree Regressor Model

```python
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
```

---

### 15. Made Predictions Using Decision Tree Regressor

```python
dt_pred = dt_model.predict(X_test)
```

---

### 16. Evaluated Decision Tree Regressor Model

```python
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_pred)

print("Decision Tree MAE:", dt_mae)
print("Decision Tree MSE:", dt_mse)
print("Decision Tree RMSE:", dt_rmse)
print("Decision Tree R2 Score:", dt_r2)
```

---

### 17. Compared Both Models

```python
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree Regressor"],
    "MAE": [lr_mae, dt_mae],
    "RMSE": [lr_rmse, dt_rmse],
    "R2 Score": [lr_r2, dt_r2]
})

comparison
```

---

## Model Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Shows the average difference between actual and predicted prices |
| MSE | Shows the average squared error |
| RMSE | Shows the prediction error in price units |
| R2 Score | Shows how well the model fits the data |

---

## Visualizations Used

The project includes the following visualizations:

- Correlation heatmap
- Actual vs predicted price graph
- Regression line plot
- Model comparison graph

These visualizations helped me understand the model performance clearly.

---

## Results

The machine learning models were able to predict car selling prices based on the given features.

Linear Regression helped in understanding the basic relationship between features and price, while Decision Tree Regressor helped capture non-linear patterns in the dataset.

The model with the higher R2 Score and lower error values performed better.

---

## Conclusion

In this project, I successfully completed a car price prediction task using machine learning.

I learned how to prepare data, create new features, encode categorical variables, train regression models, and evaluate their performance.

This project helped me understand how machine learning can be used in real-life price prediction problems.

---

## Folder Structure

```text
Task-3-Car-Price-Prediction/
│
├── car_price_prediction.ipynb
├── car data.csv
├── README.md
└── requirements.txt
```

---

## How to Run the Project

1. Download or clone this repository.

2. Open the project folder.

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Open the Jupyter Notebook file:

```text
car_price_prediction.ipynb
```

5. Run all the cells step by step.

---

## requirements.txt

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## Internship Details

**Internship:** Oasis Infobyte Data Science Internship  
**Task:** Task 3 - Car Price Prediction  
**Domain:** Data Science  
**Programming Language:** Python  
**Tool Used:** Jupyter Notebook  

---

## Author

**Bhanu Prakash Nambari**

