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

# import pandas as pd
# import numpy as np
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn import metrics

# # Step 1: Load the dataset
# data = pd.read_csv("E:/Dataset/demand.csv")  # Update path if needed

# # Step 2: Handle missing values
# data.dropna(inplace=True)

# # Step 3: Feature Engineering
# # Convert date if present
# if 'Date' in data.columns:
#     data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
#     data['Month'] = data['Date'].dt.month
#     data['DayOfWeek'] = data['Date'].dt.dayofweek

# # Create new features
# data['Effective_Price'] = data['Base Price'] * (1 - data['Discount'] / 100)
# data['Profit_Margin'] = data['Base Price'] - data['Total Price']

# # Step 4: Encode categorical features (if any exist)
# cat_columns = data.select_dtypes(include=['object']).columns.tolist()
# if cat_columns:
#     data = pd.get_dummies(data, columns=cat_columns, drop_first=True)

# # Step 5: Define features and target
# target = 'Units Sold'
# exclude_cols = ['Units Sold', 'Date'] if 'Date' in data.columns else ['Units Sold']
# X = data.drop(columns=exclude_cols)
# y = data['Units Sold']

# # Step 6: Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# # Step 7: Train the optimized RandomForestRegressor
# model = RandomForestRegressor(
#     n_estimators=200,
#     max_depth=10,
#     min_samples_split=4,
#     min_samples_leaf=2,
#     random_state=42
# )
# model.fit(X_train, y_train)

# # Step 8: Predict and evaluate
# y_pred = model.predict(X_test)

# print('📊 Model Performance:')
# print('R2:', metrics.r2_score(y_test, y_pred))
# print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# print('MSE:', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# # Step 9: Visualize Actual vs Predicted
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='b')
# plt.xlabel('Actual Units Sold')
# plt.ylabel('Predicted Units Sold')
# plt.title('Actual vs. Predicted Units Sold')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Step 10: Feature Importance
# importances = pd.Series(model.feature_importances_, index=X.columns)
# importances.sort_values().plot(kind='barh', figsize=(10, 6), title='Feature Importance')
# plt.tight_layout()
# plt.show()
