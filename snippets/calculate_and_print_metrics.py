from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Assuming y_true and y_pred are your true values and predictions
y_true = [...]  # Actual target values
y_pred = [...]  # Model predictions

# Mean Squared Error
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse}")

# Root Mean Squared Error
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Mean Absolute Error
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error: {mae}")

# R-squared
r2 = r2_score(y_true, y_pred)
print(f"R-squared: {r2}")
