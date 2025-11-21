import json
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

#import data
X_train = np.load('split_data/X_train.npy')
y_train = np.load('split_data/y_train.npy')
X_test = np.load('split_data/X_test.npy')
y_test = np.load('split_data/y_test.npy')

# initialize model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# save model
pickle.dump(lr_model, open('lr_model.pkl','wb'))

# predict on test set
lr_pred = lr_model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, lr_pred)
precision = precision_score(y_test, lr_pred)
recall = recall_score(y_test, lr_pred)


# save metrics
lr_metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall
}

with open('lr_metrics.json', 'w') as file:
    json.dump(lr_metrics_dict, file, indent=4)