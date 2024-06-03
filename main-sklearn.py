# Import the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([1.2, 2.3, 3.5, 4.1, 5.6, 6.3, 7.8, 8.1, 9.0, 10.2])

# Function to train the model
def train(x, y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x, y)
    return model

# Train the model
model = train(x, y)

# Make a prediction for a new value
x_new = 23.0
y_new = model.predict([[x_new]])
print(f"Predicted value for {x_new}: {y_new[0]}")
