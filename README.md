# CEO Salary Prediction Models 💼📊

This project demonstrates the use of various regression models to predict CEO salaries based on different features such as education, experience, and company size. The models included in the analysis are Linear Regression, Polynomial Regression, Support Vector Regression (SVR), Decision Tree Regressor, and Random Forest Regressor.

---

## Table of Contents 📑

- [Installation 🔧](#installation)
- [Dependencies 📦](#dependencies)
- [Usage 🚀](#usage)
- [Models 🧑‍🏫](#models)
- [Example 💡](#example)
- [License 📜](#license)

---

## Installation 🔧

To get started, clone the repository and install the required dependencies:

1. Clone the repository:
  ```bash
  git clone https://github.com/yourusername/CEO-Salary-Prediction.git
  ```
2.Navigate to the project folder:
  ```bash
  cd CEO-Salary-Prediction
  ```
3.Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
Dependencies 📦
This project requires the following Python libraries:
-numpy
-pandas
-scikit-learn
-statsmodels
To install all dependencies, use:
  ```bash
  pip install -r requirements.txt
  ```
Usage 🚀
To use the models and start predicting CEO salaries:

Prepare your dataset in CSV format (example: veriler.csv), ensuring it contains columns like "unvan" (title), education, experience, and company size.

Run the script ceo_salary_prediction.py to execute the regression models and view the results.

Example command:
  ```bash
  python ceo_salary_prediction.py
  ```
Models 🧑‍🏫
This project implements the following regression models to predict CEO salaries:

1. Linear Regression 📈
A basic model that assumes a linear relationship between the features and the target.

2. Polynomial Regression 🌀
An extension of linear regression that can capture non-linear relationships.

3. Support Vector Regression (SVR) 🔮
Uses kernel tricks to fit the data in high-dimensional space.

4. Decision Tree Regressor 🌳
A model that splits the data into branches based on feature values.

5. Random Forest Regressor 🌲
An ensemble of decision trees to improve prediction accuracy.

Each model is evaluated using the Ordinary Least Squares (OLS) method from the statsmodels library.

Example 💡
Here’s an example of how to use the trained models to predict CEO salaries for new data:

  ```bash
  yeni_veri = [[10, 100, 10]]  # Example data: experience, company size, and education
  yeni_veri_scaled = sc1.transform(yeni_veri)
  tahmin_svr = svr_reg.predict(yeni_veri_scaled)
  print("CEO Tahmin (SVR):", sc2.inverse_transform(tahmin_svr.reshape(-1, 1)))
  ```
This code predicts the CEO's salary based on the given input values using the trained Support Vector Regression model.




