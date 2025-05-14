import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

data = pd.read_csv("E:\Dataset\ProductDemand.csv")
data.head()

#The Data Contains Missing values
data.isnull().sum()

data = data.dropna()
fig = px.scatter(data, x="Total Price", y="Units Sold",size='Units Sold')
fig.show()
correlations = data.corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()
# Define the features (X) and target variable (y)
X = data[['Store ID', 'Total Price', 'Base Price']]
y = data['Units Sold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
feature_names = X.columns
# Create and train a linear regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Calculate the Mean Squared Error (MSE) to evaluate the model
print('R2:',metrics.r2_score(y_test,y_pred))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Visualize the predictions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Units Sold')
plt.ylabel('Predicted Units Sold')
plt.title('Actual vs. Predicted Units Sold')
plt.show()


