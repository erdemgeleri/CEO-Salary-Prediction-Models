CEO Salary Prediction Models
This project demonstrates the use of various regression models to predict CEO salaries based on different features such as education, experience, and company size. The models included in the analysis are Linear Regression, Polynomial Regression, Support Vector Regression (SVR), Decision Tree Regressor, and Random Forest Regressor.

Table of Contents
Installation
Dependencies
Usage
Models
Example
License
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/CEO-Salary-Prediction.git
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Dependencies
This project uses the following Python libraries:

numpy
pandas
scikit-learn
statsmodels
You can install all dependencies using:

bash
Copy code
pip install -r requirements.txt
Usage
Prepare your dataset in CSV format (example: veriler.csv), ensuring it contains columns like "unvan" (title), education, experience, and company size.

Run the script ceo_salary_prediction.py to execute the regression models and view the results.

Example command:

bash
Copy code
python ceo_salary_prediction.py
Models
This project implements the following regression models to predict CEO salaries:

Linear Regression: A basic model that assumes a linear relationship between the features and the target.
Polynomial Regression: An extension of linear regression that can capture non-linear relationships.
Support Vector Regression (SVR): Uses kernel tricks to fit the data in high-dimensional space.
Decision Tree Regressor: A model that splits the data into branches based on feature values.
Random Forest Regressor: An ensemble of decision trees to improve prediction accuracy.
Each model is evaluated using the Ordinary Least Squares (OLS) method from the statsmodels library.

Example
The script includes an example of how to use the trained models to predict CEO salaries for new data:

python
Copy code
yeni_veri = [[10, 100, 10]]  # Example data: experience, company size, and education
yeni_veri_scaled = sc1.transform(yeni_veri)
tahmin_svr = svr_reg.predict(yeni_veri_scaled)
print("CEO Tahmin (SVR):", sc2.inverse_transform(tahmin_svr.reshape(-1, 1)))
This code predicts the CEO's salary based on the given input values using the trained Support Vector Regression model.
