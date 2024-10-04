from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from network.network import Network
from layers.FCLayer import FCLayer
from layers.activation_layer import ActivationLayer

# Fetch the bank marketing dataset
bank_marketing = fetch_ucirepo(id=222)
X = pd.DataFrame(bank_marketing.data.features, columns=bank_marketing.metadata.feature_names)
y = pd.DataFrame(bank_marketing.data.targets, columns=['y'])

# Convert target variable to binary
y = np.where(y['y'] == 'yes', 1, 0)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

X_preprocessed = preprocessor.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define loss functions
def loss(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2

def loss_prime(y_true, y_pred):
    return y_true - y_pred

# Create and train the neural network
network = Network()
input_size = x_train.shape[1]

network.add(FCLayer((1, input_size), (1, 64)))
network.add(ActivationLayer((1, 64), (1, 64), relu, relu_prime))
network.add(FCLayer((1, 64), (1, 1)))
network.add(ActivationLayer((1, 1), (1, 1), sigmoid, sigmoid_prime))

network.setup_loss(loss, loss_prime)

# Reshape y_train to match the expected shape
y_train = y_train.reshape(-1, 1)
network.fit(x_train, y_train, epochs=100, learning_rate=0.05)

# Save and load the model
network.save('model.pkl')
network = network.load('model.pkl')

# Make predictions
y_pred = network.predict(x_test)
y_pred = np.array(y_pred).flatten()
y_pred = np.where(y_pred > 0.5, 1, 0)
y_test = np.array(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Get a random sample for new_client
random_index = random.randint(0, len(X) - 1)
new_client = X.iloc[random_index].values.reshape(1, -1)  # Reshape to 2D array

# Create DataFrame for the new client
new_client_df = pd.DataFrame(new_client, columns=X.columns)

# Ensure all categorical columns are present
for col in categorical_cols:
    if col not in new_client_df.columns:
        new_client_df[col] = 'unknown'

# Preprocess the new client data
new_client_preprocessed = preprocessor.transform(new_client_df)

# Make a prediction for the new client
prediction = network.predict(new_client_preprocessed)
predicted_class = 'yes' if prediction[0] > 0.5 else 'no'
print(f"Prediction for new client: {predicted_class}")
