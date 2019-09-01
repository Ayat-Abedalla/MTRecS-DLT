# -*- coding: utf-8 -*-

from preprocess import *
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.externals import joblib 


# Load and prepare the data
train_data, test_data = load_prepare_data()
# Generate the features
trainX, y_train, testX, y_test = get_prepare_data(train_data, test_data)

# Save data
save_data(trainX, y_train, testX, y_test)

# Load saved data
trainX, y_train, testX, y_test = load_data()

# Normalize context features
merged_train, X_test = build_norm_context(trainX, testX)

# Split train into train set and validation set
X_train, X_val, train_labels, val_labels = train_test_split(merged_train, y_train, test_size=0.2, random_state=42)

xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8, objective='multi:softmax')

# Train Xgboost classifier
xgb.fit(X_train.values, train_labels.values.ravel())

# Evaluate model on validation set
pred_val = xgb.predict(X_val.values)
score_val = f1_score(val_labels, pred_val, average='weighted')
print(score_val)
pred_val_prob = xgb.predict_proba(X_val.values)
np.savetxt('submit_xgb/pred_val_prob.csv',pred_val_prob ,delimiter=',')

# Make prediction for the test set
test_pred = xgb.predict(X_test.values)
pred_test_prob = xgb.predict_proba(X_test.values)
y_test_ensemble = y_test
y_test['recommend_mode'] = test_pred
y_test.to_csv('submit_xgb/test_result.csv', index=False)
np.savetxt('submit_xgb/pred_test_prob.csv',pred_test_prob ,delimiter=',')
  
# Save the model as a pickle in a file 
joblib.dump(xgb, 'xgb_output/xgb.pkl') 

# Load the model from the file 
xgb_from_pickle = joblib.load('xgb_output/xgb.pkl')  

# Weighted Average Ensemble
def ensemble_predictions(first_filename, second_filename, prob_filename):
  first_model_test = np.genfromtxt(first_filename, delimiter=',')
  second_model_test = np.genfromtxt(second_filename, delimiter=',')
  models_pred = [first_model_test, second_model_test]
  summed_weight = np.tensordot(models_pred, [0.5, 0.5], axes=((0),(0)))
  np.savetxt(prob_filename, summed_weight ,delimiter=',')
  result = np.argmax(summed_weight, axis=1)
  return result

result_val_ensemble = ensemble_predictions('submit_cnn/val_prediction.csv','submit_xgb/pred_val_prob.csv','submit_xgb/cnn_xgb_val.csv')
score = f1_score(val_labels, result_val_ensemble, average='weighted')
print(score)

result_test_ensemble = ensemble_predictions('submit_cnn/prediction.csv', 'submit_xgb/pred_test_prob.csv', 'submit_xgb/cnn_xgb_test_pred.csv')
y_test_ensemble['recommend_mode'] = result_test_ensemble
y_test_ensemble.to_csv('submit_xgb/cnn_xgb_test_result.csv', index=False)
