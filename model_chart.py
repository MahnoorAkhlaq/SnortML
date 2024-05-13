import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the dataset
dataset_path = "cleaned_data.csv"
data = pd.read_csv(dataset_path)

# Preprocessing: Drop irrelevant columns and handle missing values
data.dropna(inplace=True)
X = data.drop(['Timestamp', 'SensorId', 'SourceIP'], axis=1)
y = data['sus']  # Assuming 'sus' column indicates suspicious activity (label)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM input (batch_size, timesteps, input_dim)
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the RNN model
model = Sequential([
    LSTM(units=128, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dense(units=64, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=16, activation='relu'),
    Dense(units=8, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define model checkpoint to save the best model during training
checkpoint = ModelCheckpoint("rnn_attack_classifier_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=64, validation_data=(X_test_reshaped, y_test), callbacks=[checkpoint])

# Save model architecture as an image
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Plot training history
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Save the figure
plt.savefig('training_history.png')

# Show plot
plt.show()

# Get the weights of the LSTM layer
lstm_layer = model.layers[0]
weights = lstm_layer.get_weights()[0]

# Compute the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(weights.squeeze(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('LSTM Layer Weights Heatmap')
plt.xlabel('Hidden Units')
plt.ylabel('Input Features')
plt.savefig('lstm_weights_heatmap.png')
plt.show()
