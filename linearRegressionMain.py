# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("sample_data.csv")
data.fillna(0, inplace=True)
# Selecting relevant features
# ID,Severity,Start_Lat,Start_Lng,StartTime,EndTime,Distance(mi),DelayFromTypicalTraffic(mins),DelayFromFreeFlowSpeed(mins),Congestion_Speed,Description,Street,City,County,State,Country,ZipCode,LocalTimeZone,WeatherStation_AirportCode,WeatherTimeStamp,Temperature(F),WindChill(F),Humidity(%),Pressure(in),Visibility(mi),WindDir,WindSpeed(mph),Precipitation(in),Weather_Event,Weather_Conditions
X = data[['Start_Lat','Start_Lng','Distance(mi)','DelayFromTypicalTraffic(mins)','DelayFromFreeFlowSpeed(mins)','Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'WindSpeed(mph)']]
y = data['Severity']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predicting the severity for a given set of features
new_data = [[42.410881, -71.147995, 3.990000009536743, 0.0, 4.0, 64.0, 63.0, 29.93, 7.0]]  # Example features for a specific instance
predicted_severity = model.predict(new_data)
print("Predicted Severity:", predicted_severity)