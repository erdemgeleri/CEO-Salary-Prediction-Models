import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

veriler = pd.read_csv('veriler.csv')
label_encoder = LabelEncoder()
veriler['unvan'] = label_encoder.fit_transform(veriler['unvan'])
x = veriler.iloc[:, 2:5]  
y = veriler.iloc[:, 5:6]  
X = x.values
Y = y.values
print('-------------------------\n\n')
print("Korelasyon Matrisi:")
print(veriler.corr())
print('-------------------------\n\n')

# Linear
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
model = sm.OLS(Y, sm.add_constant(X))  
print("Linear Regression OLS Modeli Özeti:")
print(model.fit().summary())
print('-------------------------\n\n')



# Poly
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)
print('Polinom OLS Modeli Özeti:')
model2 = sm.OLS(Y, sm.add_constant(X_poly))
print(model2.fit().summary())
print('-------------------------\n\n')



# SVR  
sc1 = StandardScaler()
X_scaled = sc1.fit_transform(X)  
sc2 = StandardScaler()
Y_scaled = sc2.fit_transform(Y)  
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_scaled, Y_scaled)
print('SVR OLS Modeli Özeti:')
model3 = sm.OLS(Y_scaled, sm.add_constant(X_scaled))
print(model3.fit().summary())
print('-------------------------\n\n')



# Decision Tree Regresyon
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X, Y)
print('Decision Tree OLS Modeli Özeti:')
model4 = sm.OLS(Y, sm.add_constant(X))
print(model4.fit().summary())
print('-------------------------\n\n')



# Random Forest Regresyon
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y.ravel())
print('Random Forest OLS Modeli Özeti:')
model5 = sm.OLS(Y, sm.add_constant(X))
print(model5.fit().summary())
print('-------------------------\n\n')



#ORNEK
yeni_veri = [[10, 100, 10]] 
yeni_veri_scaled = sc1.transform(yeni_veri) 
tahmin_svr = svr_reg.predict(yeni_veri_scaled)  
print("CEO Tahmin (SVR):", sc2.inverse_transform(tahmin_svr.reshape(-1, 1)))
tahmin_dt = r_dt.predict(yeni_veri)
print('-------------------------\n\n')
print("CEO Tahmin (Decision Tree):", tahmin_dt)
tahmin_rf = rf_reg.predict(yeni_veri)
print('-------------------------\n\n')
print("CEO Tahmin (Random Forest):", tahmin_rf)
print('-------------------------\n\n')
