# Task 3: Car Price Prediction Using Machine Learning

## Project Overview

This project is part of my **Oasis Infobyte Data Science Internship**.

The aim of this project is to build a machine learning model that predicts the **selling price of a used car** based on different features such as car age, present price, kilometers driven, fuel type, selling type, transmission type, owner details, and brand.

Car price prediction is a regression-based machine learning problem because the target value, `Selling_Price`, is a continuous numerical value.

---

## Objective

The main objective of this project is to predict the selling price of a used car using machine learning regression models.

Through this project, I learned how to:

- Load and understand a dataset
- Clean and preprocess data
- Handle missing and duplicate values
- Fix inconsistent categorical values
- Create new features from existing columns
- Extract brand names from car names
- Perform exploratory data analysis
- Encode categorical variables
- Train and compare regression models
- Evaluate model performance using regression metrics
- Visualize important features affecting car price

---

## Dataset Information

The dataset contains information about used cars and their selling prices.

### Dataset Columns

| Column Name | Description |
|---|---|
| Car_Name | Name of the car |
| Year | Year of purchase |
| Selling_Price | Price at which the car is being sold |
| Present_Price | Current showroom price of the car |
| Driven_kms | Total kilometers driven by the car |
| Fuel_Type | Type of fuel used by the car |
| Selling_type | Whether the seller is a dealer or individual |
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

In this project, I used two regression models:

1. **Linear Regression**
2. **Decision Tree Regressor**

These models were used to compare prediction performance. I used these models instead of Random Forest because Random Forest was already used in a previous task.

---

## Project Workflow

### 1. Importing Required Libraries

The required Python libraries were imported for data analysis, visualization, and machine learning.

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

### 2. Loading the Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv("car data.csv")
```

---

### 3. Understanding the Dataset

The dataset was explored using basic Pandas functions.

```python
df.head()
df.shape
df.info()
df.describe()
```

This helped in understanding the number of rows, columns, data types, and basic statistics of the dataset.

---

### 4. Checking Missing Values

Missing values were checked using:

```python
df.isnull().sum()
```

This step helped confirm whether any null values were present in the dataset.

---

### 5. Removing Duplicate Values

Duplicate records were checked and removed.

```python
df.duplicated().sum()
df = df.drop_duplicates()
```

Removing duplicates helps improve data quality and avoids repeated records affecting the model.

---

### 6. Cleaning Categorical Values

Categorical columns were cleaned by removing extra spaces and standardizing text format.

```python
df["Fuel_Type"] = df["Fuel_Type"].str.strip().str.title()
df["Selling_type"] = df["Selling_type"].str.strip().str.title()
df["Transmission"] = df["Transmission"].str.strip().str.title()
```

This step helps avoid inconsistencies such as `Petrol`, `petrol`, or extra spaces being treated as different values.

---

### 7. Feature Engineering

A new column called `Car_Age` was created from the `Year` column.

```python
df["Car_Age"] = 2018 - df["Year"]
```

This feature helps the model understand how old the car is.

---

### 8. Brand Extraction

The brand name was extracted from the `Car_Name` column.

```python
df["Brand"] = df["Car_Name"].str.split().str[0]
```

This helps include brand information in the prediction model.

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the relationship between different features and the selling price.

### Selling Price Distribution

A histogram was used to understand the distribution of selling prices.

```python
plt.figure(figsize=(8, 5))
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Distribution of Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Count")
plt.show()
```

---

### Selling Price by Fuel Type

A box plot was used to compare selling prices across different fuel types.

```python
plt.figure(figsize=(8, 5))
sns.boxplot(x="Fuel_Type", y="Selling_Price", data=df)
plt.title("Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
plt.show()
```

---

### Selling Price vs Car Age

A scatter plot was used to analyze the relationship between car age and selling price.

```python
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Car_Age", y="Selling_Price", data=df)
plt.title("Selling Price vs Car Age")
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
plt.grid(True)
plt.show()
```

From this graph, it can be observed that newer cars usually have higher selling prices, while older cars usually have lower selling prices.

---

### Correlation Heatmap

A correlation heatmap was used to understand the relationship between numerical features.

```python
numeric_columns = ["Selling_Price", "Present_Price", "Driven_kms", "Owner", "Car_Age"]

plt.figure(figsize=(8, 5))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()
```

---

## Data Preprocessing

### Dropping Unnecessary Columns

After extracting useful information, unnecessary columns were removed.

```python
df = df.drop(["Car_Name", "Year"], axis=1)
```

---

### Encoding Categorical Variables

Categorical variables were converted into numerical format using one-hot encoding.

```python
df = pd.get_dummies(df, drop_first=True)
```

This made the dataset suitable for machine learning models.

---

### Selecting Features and Target

The target variable was `Selling_Price`.

```python
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
```

---

### Train-Test Split

The dataset was split into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Model Training and Evaluation

### Linear Regression

Linear Regression was trained as the first regression model.

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
```

The model was evaluated using MAE, RMSE, and R2 Score.

```python
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression MAE:", lr_mae)
print("Linear Regression RMSE:", lr_rmse)
print("Linear Regression R2 Score:", lr_r2)
```

---

### Decision Tree Regressor

Decision Tree Regressor was trained as the second regression model.

```python
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
```

The model was evaluated using the same regression metrics.

```python
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_pred)

print("Decision Tree MAE:", dt_mae)
print("Decision Tree RMSE:", dt_rmse)
print("Decision Tree R2 Score:", dt_r2)
```

---

## Model Comparison

The performance of both models was compared using MAE, RMSE, and R2 Score.

```python
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree Regressor"],
    "MAE": [lr_mae, dt_mae],
    "RMSE": [lr_rmse, dt_rmse],
    "R2 Score": [lr_r2, dt_r2]
})

comparison
```

The better model was selected based on lower error values and higher R2 Score.

---

## Feature Importance

Feature importance was plotted for the Decision Tree Regressor model.

```python
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance
```

```python
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10))
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
```

This chart shows which features had the highest impact on predicting car selling price.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error shows the average absolute difference between actual and predicted prices |
| RMSE | Root Mean Squared Error shows the prediction error in the same unit as the target variable |
| R2 Score | Shows how well the model explains the variation in selling price |

---

## Results

Both Linear Regression and Decision Tree Regressor were trained and evaluated.

Linear Regression helped understand the basic linear relationship between car features and selling price. Decision Tree Regressor helped capture non-linear relationships in the dataset.

The model with the lower MAE and RMSE values and the higher R2 Score was considered the better-performing model.

---

## Conclusion

In this project, I successfully built a machine learning model to predict used car selling prices.

The dataset was cleaned by checking missing values, removing duplicates, and standardizing categorical values. Feature engineering was performed by creating a `Car_Age` column and extracting the `Brand` from the car name. Exploratory data analysis was completed using different visualizations such as selling price distribution, price by fuel type, price vs car age, and correlation heatmap.

Two regression models, Linear Regression and Decision Tree Regressor, were trained and evaluated using MAE, RMSE, and R2 Score.

This project helped me understand how regression models can be used to solve real-world price prediction problems.

---

## Folder Structure

```text
DataScience-Task3-CarPricePrediction/
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

3. Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Open the Jupyter Notebook file.

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

---

## Acknowledgement

I would like to thank **Oasis Infobyte** for giving me this opportunity to work on practical data science projects and improve my machine learning skills.
