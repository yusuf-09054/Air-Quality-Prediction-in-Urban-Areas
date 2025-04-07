
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming model, X, X_test, y_test, y_pred, and y are already defined

# 6. Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# 7. Graph: Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='white', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs Predicted AQI (24h Forecast)')
plt.grid(True)
plt.axis('equal')
plt.text(0.05, 0.95, f'R² = {r2:.2f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
plt.tight_layout()
plt.show()

# 8. Graph: Feature Importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]  # descending order

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Feature Importance in AQI Prediction')
plt.gca().invert_yaxis()  # most important feature on top
plt.tight_layout()
plt.show()
