import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('boros_energy_matches.csv')

# Convert date to numeric (days since first match)
df['date'] = pd.to_datetime(df['date'])
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Encode categorical features
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['opponent_archetype', 'event_type']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

# Combine features
X = pd.concat([df[['days_since_start']], encoded_df], axis=1)
y = df['winrate']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Plot predictions
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Winrate")
plt.ylabel("Predicted Winrate")
plt.title("Boros Energy Winrate Prediction")
plt.show()