from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from network.network import Network
from layers.FCLayer import FCLayer
from layers.activation_layer import ActivationLayer

bank_marketing = fetch_ucirepo(id=222)
X = pd.DataFrame(bank_marketing.data.features, columns=bank_marketing.metadata.feature_names)
y = pd.DataFrame(bank_marketing.data.targets, columns=['y'])

y = np.where(y == 'yes', 1, 0)

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X_preprocessed = preprocessor.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    x[x<0] = 0
    x[x>0] = 1
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def loss(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2

def loss_prime(y_true, y_pred):
    return y_true - y_pred

network = Network()

input_size = x_train.shape[1]

network.add(FCLayer((1, input_size), (1, 64)))
network.add(ActivationLayer((1, 64), (1, 64), relu, relu_prime))
network.add(FCLayer((1, 64), (1, 1)))
network.add(ActivationLayer((1, 1), (1, 1), sigmoid, sigmoid_prime))

network.setup_loss(loss, loss_prime)

y_train = y_train.reshape(-1, 1)
network.fit(x_train, y_train, epochs=100, learning_rate=0.05)

network.save('model.pkl')

network = network.load('model.pkl')

y_pred = network.predict(x_test)

y_pred = np.where(y_pred > 0.5, 1, 0)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

new_client = np.array([[30, 'technician', 'married', 'secondary', 6000, 'no', 'no', 'cellular', 'may', 180, 2, 999, 1, 'failure', 'nonexistent']])

new_client_preprocessed = preprocessor.transform(new_client)

prediction = network.predict(new_client_preprocessed)
predicted_class = 'yes' if prediction > 0.5 else 'no'
print(f"Prediction: {predicted_class}")