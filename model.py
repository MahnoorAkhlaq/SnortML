import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Suppress TensorFlow information messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the dataset
df = pd.read_csv("cleaned_data.csv")

# Selecting required columns
# selected_columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'Label']
# df = df[selected_columns]

# Drop rows with missing values
df.dropna(inplace=True)

# Splitting data into features (X) and target variable (y)
X = df[['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state']]
y = df['Label']

# Apply feature hashing to 'srcip', 'dstip', 'sport', and 'dsport' columns
def feature_hasher(data):
    return pd.DataFrame(data.apply(lambda x: [hash(str(val)) % 1000 for val in x]))

X_hashed = feature_hasher(X)

# Encode 'proto' and 'state' columns
label_encoder = LabelEncoder()
X_encoded = X.copy()
for col in ['proto', 'state']:
    X_encoded[col] = label_encoder.fit_transform(X[col])

# Concatenate hashed features with encoded features
X_processed = pd.concat([X_hashed, X_encoded], axis=1)

# Remove non-numeric columns
X_numeric = X_processed.select_dtypes(include=np.number)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM input (batch_size, timesteps, input_dim)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the RNN model
model = Sequential([
    LSTM(units=128, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True),
    Dropout(0.2),  # Add dropout for regularization
    LSTM(units=64),
    Dense(units=32, activation='relu'),
    Dropout(0.2),  # Add dropout for regularization
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define model checkpoint to save the best model during training
checkpoint = ModelCheckpoint("rnn_attack_classifier_model_updated.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=128, validation_data=(X_test_reshaped, y_test), callbacks=[checkpoint])

# Evaluate the model
y_pred_probs = model.predict(X_test_reshaped)
y_pred = (y_pred_probs > 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
