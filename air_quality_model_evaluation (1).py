import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📌 Step 1: Generate a Sample Dataset
np.random.seed(42)

data_size = 500  # Number of rows
data = {
    "PM2.5": np.random.randint(10, 150, data_size),
    "PM10": np.random.randint(20, 200, data_size),
    "NO2": np.random.randint(5, 80, data_size),
    "CO": np.round(np.random.uniform(0.1, 2.5, data_size), 2),
    "Temperature": np.random.randint(15, 40, data_size),
    "Humidity": np.random.randint(30, 90, data_size),
    "Wind Speed": np.round(np.random.uniform(0.5, 5.0, data_size), 2),
    "AQI": np.random.randint(50, 300, data_size)  # Target Variable
}

df = pd.DataFrame(data)

# 📌 Step 2: Define Features and Target Variable
X = df.drop(columns=["AQI"])  # Features
y = df["AQI"]  # Target variable

# 📌 Step 3: Split Data into Training and Testing Sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 4: Train the Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Step 5: Make Predictions
y_pred = model.predict(X_test)

# 📌 Step 6: Evaluate Model Performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 📌 Step 7: Print Results
print("✅ Model Performance Evaluation:")
print(f"🔹 RMSE: {rmse:.2f}")
print(f"🔹 MAE: {mae:.2f}")
print(f"🔹 R² Score: {r2:.4f}")
