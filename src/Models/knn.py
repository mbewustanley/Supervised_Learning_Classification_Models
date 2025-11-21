import json
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

#import data
X_train = np.load('split_data/X_train.npy')
y_train = np.load('split_data/y_train.npy')
X_test = np.load('split_data/X_test.npy')
y_test = np.load('split_data/y_test.npy')

# initialize model
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)

# save model
pickle.dump(knn_model, open('knn_model.pkl','wb'))

# predict on test set
knn_pred = knn_model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, knn_pred)
precision = precision_score(y_test, knn_pred)
recall = recall_score(y_test, knn_pred)


# save metrics
knn_metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall
}

with open('knn_metrics.json', 'w') as file:
    json.dump(knn_metrics_dict, file, indent=4)