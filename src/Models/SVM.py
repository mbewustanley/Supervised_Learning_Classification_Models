import json
import numpy as np
import pickle

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

#import data
X_train = np.load('split_data/X_train.npy')
y_train = np.load('split_data/y_train.npy')
X_test = np.load('split_data/X_test.npy')
y_test = np.load('split_data/y_test.npy')

# initialize model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# save modsvm
pickle.dump(svm_model, open('svm_model.pkl','wb'))

# predict on test set
svm_pred = svm_model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, svm_pred)
precision = precision_score(y_test,svm_pred)
recall = recall_score(y_test, svm_pred)


# save metrics
svm_metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall
}

with open('svm_metrics.json', 'w') as file:
    json.dump(svm_metrics_dict, file, indent=4)