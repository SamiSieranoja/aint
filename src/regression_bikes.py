import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from pathlib import Path
# pip install pandas scikit-learn numpy

# --- Settings ---
# These two rows will be kept out of training and used to test predictions
test_instances = [645, 9332]
# The column we want to predict
target = "cnt"
# The columns we use as inputs to the model
features = ["temp", "hr", "hum", "weekday"]

# --- Load data ---
# Data from here (CC BY 4.0): https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
df = pd.read_csv(Path(__file__).parent.parent / "data" / "bike_sharing.csv")
# Remove any rows that have missing values in the columns we need
df = df.dropna(subset=features + [target])

# --- Split into train and test ---
# Create a True/False mask: True for rows whose instant is in test_instances
test_mask = df["instant"].isin(test_instances)
df_train = df[~test_mask]   # ~ means NOT, so this is all rows except test
df_test = df[test_mask]     # just the two test rows

# Extract feature matrix X and target vector y as plain numpy arrays
X_train = df_train[features].values
y_train = df_train[target].values
X_test_raw = df_test[features].values
y_test = df_test[target].values

# --- Normalize features ---
# Linear regression works better when all features are on the same scale.
# StandardScaler transforms each feature to have mean=0 and std=1:
#   x_normalized = (x - mean) / std
# fit_transform: learns the mean/std from training data and applies it
# transform: applies the same mean/std to test data (do NOT refit on test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test_raw)

# --- Train model ---
# LinearRegression finds the coefficients (slopes) and intercept that
# minimize the sum of squared errors between predicted and actual values
model = LinearRegression()
model.fit(X_train, y_train)  # this is where the learning happens

# --- Evaluate on training data ---
# model.predict() applies: y = c + m1*x1 + m2*x2 + m3*x3 + m4*x4
y_pred = model.predict(X_train)
# SSE: sum of squared errors, total squared error across all training rows
sse = np.sum((y_train - y_pred) ** 2)

print(f"Linear regression: {target}")
print(f"N = {len(y_train)}, SSE = {sse:.2f}\n")

# --- Print coefficients ---
# A positive coefficient means the feature increases the predicted cnt.
# A negative coefficient means it decreases it.
# Because features are normalized, coefficients are directly comparable.
print(f"{'Feature':<20} {'':>4} {'Coef':>9}")
print("-" * 35)
for j, (feat, coef) in enumerate(zip(features, model.coef_), start=1):
    print(f"{feat:<20} m{j:<3} {coef:>9.4f}")
print("-" * 35)
print(f"{'Intercept':<20} c    {model.intercept_:>9.4f}")

# --- Predict test instances ---
# prediction = c + m1*x1 + m2*x2 + m3*x3 + m4*x4
print("\nPredictions:")
print(f"{'Instant':<10} {'Actual':>9}  {'Predicted':>9}")
print("-" * 32)
for i, instant in enumerate(test_instances):
    x = X_test[i]
    pred = (model.intercept_
            + model.coef_[0] * x[0]   # m1 * temp
            + model.coef_[1] * x[1]   # m2 * hr
            + model.coef_[2] * x[2]   # m3 * hum
            + model.coef_[3] * x[3])  # m4 * weekday
    print(f"{instant:<10} {y_test[i]:>9.1f}  {pred:>9.4f}")
    
    
    
    
