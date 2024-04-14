# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv("sample_data.csv")
data.fillna(0, inplace=True)

# Selecting relevant features
X = data[['Start_Lat','Start_Lng','Distance(mi)','DelayFromTypicalTraffic(mins)','DelayFromFreeFlowSpeed(mins)','Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'WindSpeed(mph)']]
y = data['Severity']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the features
X_train_norm = tf.keras.utils.normalize(X_train, axis=1)
X_test_norm = tf.keras.utils.normalize(X_test, axis=1)

# Building the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_norm.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer with single neuron for severity prediction
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train_norm, y_train, epochs=10, batch_size=500, validation_split=0.2)

# Evaluating the model

# Calculating mean squared error
mse = model.evaluate(X_test_norm, y_test)
print("Mean Squared Error:", mse)

# Calculating mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Calculating R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)
# Saving the modelp
model.save("severity_prediction_model.h5")

# Loading the saved model
loaded_model = tf.keras.models.load_model("severity_prediction_model.h5")

# Predicting severity for new test data
new_test_data = [[43.037544, -71.321381, 1.8700000047683718, 8.0, 7.0, 67.0, 59.0, 29.8, 3.0]]  # Example features for a specific instance
new_test_data_norm = tf.keras.utils.normalize(new_test_data, axis=1)
predicted_severity = loaded_model.predict(new_test_data_norm)
print("Predicted Severity:", predicted_severity)
