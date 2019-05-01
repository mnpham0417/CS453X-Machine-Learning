'''
Accuracy for linear SVC: 0.8587113790667549
Total time for linear SVC: 1.7014569640159607 minutes

Accuracy for polynomial SVC: 0.8421354311965332
Total time for polynomial SVC: 15.53552513519923 minutes
'''

import pandas
import numpy as np
import time, math
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import roc_auc_score

def measureTime(start, end):
    return (end - start)

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

# Split into train/test folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify = y)

# Linear SVM
yhat1 = np.zeros((len(y_test)))
start1 = time.time()
linear_svc = LinearSVC(dual=False)
linear_svc.fit(X_train,y_train)
linear_svc_pred = linear_svc.decision_function(X_test)
end1 = time.time()

# Non-linear SVM (polynomial kernel)
ind = 0
size = 5000
testing = 0
yhat2 = np.zeros((len(y_test)))
start2 = time.time()
for itr in range(math.ceil(len(y_train)/size)):
    poly_svc = SVC(kernel="poly", degree = 3, gamma = "auto")
    poly_svc.fit(X_train[ind:ind+size],y_train[ind:ind+size])
    poly_svc_pred = poly_svc.decision_function(X_test)
    yhat2 = yhat2 + poly_svc_pred
    ind = ind + size
end2 = time.time()

# Apply the SVMs to the test set
yhat1 = linear_svc_pred
yhat2 = yhat2/math.ceil(len(y_train)/size)

# Compute AUC
auc1 = roc_auc_score(y_test, yhat1)
auc2 = roc_auc_score(y_test, yhat2)

print(auc1)
print(auc2)

#You can comment out to see the computation time
# print("Total time for linear SVC: {} minutes".format(measureTime(start1,end1)/60))
# print("Total time for polynomial SVC: {} minutes".format(measureTime(start2,end2)/60))