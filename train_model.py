import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('C:/Users/shaili/OneDrive/Documents/Desktop/shaili/shaili/Clean_Dataset.csv')

# Drop the unnamed index column and the 'flight' column
data = data.drop(columns=['Unnamed: 0', 'flight'])

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class'])

# Select features and target
X = data.drop(columns=['price'])
y = data['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a pickle file
joblib.dump(model, 'flight_price_model.pkl')

# Save the column names
joblib.dump(X.columns, 'model_columns.pkl')
