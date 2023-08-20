from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv("sales_data.csv")
print(data.head())
print(data.info())
# Handle missing values
data.fillna(0, inplace=True)
# Create new features
data["Year"] = pd.to_datetime(data["Date"]).dt.year
data["Month"] = pd.to_datetime(data["Date"]).dt.month
# Define features and target variable
X = data[["Promotions", "Holidays", "Year", "Month"]]
y = data["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    promotions = data["promotions"]
    holidays = data["holidays"]
    year = data["year"]
    month = data["month"]

    features = np.array([[promotions, holidays, year, month]])
    prediction = model.predict(features)
    return jsonify({"prediction": prediction[0]})
if __name__ == "__main__":
    app.run(debug=True)