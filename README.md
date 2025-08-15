import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = pd.read_csv('AirQuality.csv', delimiter=';')
data = data.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1)
data.columns = [col.strip().lower() for col in data.columns]
print(data.head())
data = data.dropna()
for col in data.columns:
  if col not in ['date', 'time']:
    data[col] = data[col].astype(str).str.replace(',', '.', regex=False)
    data[col] = pd.to_numeric(data[col], errors='coerce')

sns.pairplot(data)
plt.show()

corr = data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
print(data.columns)
X = data[['co(gt)', 'pt08.s1(co)', 'c6h6(gt)', 'pt08.s2(nmhc)', 'nox(gt)', 'pt08.s3(nox)', 'no2(gt)', 'pt08.s4(no2)', 'pt08.s5(o3)', 't', 'rh', 'ah']]
y = data['pt08.s2(nmhc)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = data[['co(gt)', 'pt08.s1(co)', 'c6h6(gt)', 'pt08.s2(nmhc)', 'nox(gt)', 'pt08.s3(nox)', 'no2(gt)', 'pt08.s4(no2)', 'pt08.s5(o3)', 't', 'rh', 'ah']]
y = data['pt08.s2(nmhc)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual AQI')
plt.plot(y_pred, label='Predicted AQI', alpha=0.7)
plt.title('Actual vs Predicted AQI')
plt.legend()
plt.show()
