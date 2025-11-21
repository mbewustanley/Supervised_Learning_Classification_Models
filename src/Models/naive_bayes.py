import json
import numpy as np
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score

#import data
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')

# initialize model
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

# save model
pickle.dump(nb_model, open('nb_model.pkl','wb'))

# predict on test set
nb_pred = nb_model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, nb_pred)
precision = precision_score(y_test, nb_pred)
recall = recall_score(y_test, nb_pred)


# save metrics
nb_metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall
}

with open('nb_metrics.json', 'w') as file:
    json.dump(nb_metrics_dict, file, indent=4)