#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 00:06:06 2019

@author: ayat
"""

from preprocess import *
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization, Activation, Flatten
from keras.models import model_from_json
from keras.models import load_model  

# Load and prepare the data
train_data, test_data = load_prepare_data()
# Generate the features
trainX, y_train, testX, y_test = get_prepare_data(train_data, test_data)

# Save data
#save_data(trainX, y_train, testX, y_test)

# Load saved data
#trainX, y_train, testX, y_test = load_data()

# Normalize context features
merged_train, X_test = build_norm_context(trainX, testX)

# Split train into train set and validation set
X_train, X_val, train_labels, val_labels = train_test_split(merged_train, y_train, test_size=0.2, random_state=42)

# Prepare data for the Conv1D
X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val = X_val.values.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the model
model = Sequential()

model.add(Conv1D(32, 5, padding='same', input_shape=(1,57), name='conv1d_1'))
model.add(Activation(activation='relu', name='activation_1'))

model.add(Conv1D(64, 5, padding='same', name='conv1d_2'))
model.add(Activation(activation='relu', name='activation_2'))

model.add(Conv1D(128, 3, padding='same', name='conv1d_3'))
model.add(BatchNormalization(name='batch_normalization__1'))
model.add(Activation(activation='relu', name='activation_3'))

model.add(Conv1D(256, 3, padding='same', name='conv1d_4'))
model.add(BatchNormalization(name='batch_normalization__2'))
model.add(Activation(activation='relu', name='activation_4'))

model.add(MaxPooling1D(pool_size=2,data_format='channels_first', name='max_pooling1d_1'))

model.add(Flatten(name='flatten'))

model.add(Dense(512, name='dense1'))

model.add(Dense(12, activation='softmax', name='output'))

model.summary()

# Set Optimizer
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, train_labels, batch_size=64, shuffle=True, epochs=100)

# Evaluate model on validation set
val_loss, val_acc = model.evaluate(X_val, val_labels)
print("Val Loss: {:.5f} ".format(val_loss))
print("Val Accuracy: {:.2f}% ".format(val_acc*100))

# Make prediction for the valdation set to calculate F1-Score
predictions = model.predict(X_val)
y_pred = np.argmax(predictions, axis=1)
score = f1_score(val_labels, y_pred, average='weighted')
print("Weighted F1-score = ", score)
np.savetxt('submit_cnn/val_prediction.csv',predictions ,delimiter=',')

# Make prediction for the test set
pred = model.predict(X_test)
test_pred = np.argmax(pred, axis=1)
# Save prediction for submission
y_test['recommend_mode'] = test_pred
y_test.to_csv('submit_cnn/test_result.csv', index=False)
np.savetxt('submit_cnn/prediction.csv',pred ,delimiter=',')

# Save final model
# serialize model to JSON
model_json = model.to_json()
with open("CNN_output/final_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("CNN_output/final_model.h5")

# Load model
model.load_weights("CNN_output/final_model.h5")