import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

import os 
from pathlib import Path 
from dotenv import load_dotenv

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor


PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

REGRESSION_DIR = os.environ["REGRESSION_DIR"]

df = pd.read_csv(os.path.join(REGRESSION_DIR / "vehicles_cleaned.csv"))

X = df.drop(columns=['record_id', 'fuel_consumption_l_100km'])
y = df['fuel_consumption_l_100km']

X = pd.get_dummies(
    X, 
    columns = ['vehicle_type'],
    drop_first = True,
    dtype = int
)

imputer = SimpleImputer(strategy = 'median')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X.shape)
print(y.shape)
print(X.isna().sum())

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_predicted = lr.predict(X_test_scaled)

print('Linear Regression - metrics: ')
print('r2: ', r2_score(y_test, y_predicted))
print('mean_squarred_error: ', mean_squared_error(y_test, y_predicted))
print('rmse: ', mean_squared_error(y_test, y_predicted) ** 0.5)
print('mean_absolute_error: ', mean_absolute_error(y_test, y_predicted))

rf = RandomForestRegressor(
    n_estimators = 100, 
    random_state = 42
)

rf.fit(X_train, y_train)

y_predicted_rf = rf.predict(X_test)

print('Random forest - metrics')
print('r2: ', r2_score(y_test, y_predicted_rf))
print('mean_squarred_error: ', mean_squared_error(y_test, y_predicted_rf))
print('rmse: ', mean_squared_error(y_test, y_predicted_rf) ** 0.5)
print('mean_absolute_error: ', mean_absolute_error(y_test, y_predicted_rf))

gbr = GradientBoostingRegressor(
    n_estimators = 200, 
    learning_rate = 0.1,
    max_depth = 3, 
    random_state = 42
)

gbr.fit(X_train, y_train)

y_predicted_gbr = gbr.predict(X_test)

print('Gradient boosting')
print('r2: ', r2_score(y_test, y_predicted_gbr))
print('mean_squarred_error: ', mean_squared_error(y_test, y_predicted_gbr))
print('rmse: ', mean_squared_error(y_test, y_predicted_gbr) ** 0.5)
print('mean_absolute_error: ', mean_absolute_error(y_test, y_predicted_gbr))

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient':lr.coef_
})

coefficients['abs_coefficient'] = coefficients['Coefficient'].abs()

coefficients = coefficients.sort_values('abs_coefficient', ascending = False)
print(coefficients)

corr = X.corr(numeric_only = True)
print(corr)

print(X[['engine_size_l', 'vehicle_mass_kg', 'power_kw']].corr())

plt.figure(figsize=(12,9))
plt.imshow(corr, cmap = 'coolwarm', vmin = -1, vmax = 1)
plt.colorbar()
plt.title('Correlation')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()

vif = pd.DataFrame()
vif['Feature'] =  X.columns

vif['VIF'] = [
    variance_inflation_factor(X_train_scaled,i)
    for i in range(X_train_scaled.shape[1])
]

vif = vif.sort_values('VIF', ascending = False)
print(vif)

test_car = pd.DataFrame([{
    'engine_size_l': 1.4,
    'cylinders':4,
    'vehicle_mass_kg': 1269,
    'power_kw': 132,
    'aero_drag_coefficient':0.328,
    'avg_speed_kmh':100,
    'avg_engine_rpm':2500,
    'engine_temp_c':90,
    'tire_pressure_bar':2.3,
    'months_since_service':10,
    'road_grade_pct':0,
    'ac_on':0,
    'vehicle_type_compact':1,
    'vehicle_type_sedan':0
}])

test_car_imputed = imputer.transform(test_car)
test_car_scaled = scaler.transform(test_car_imputed)
prediction = lr.predict(test_car_scaled)
print(f"Predicted fuel consumption: {prediction[0]:.2f} L/100 km")

