import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

input_file = "income_data.txt"

X = []
y = []

for line in open(input_file):
    if "?" in line:
        continue

    data = line.strip().split(", ")

    if len(data) != 15:
        continue

    X.append(data[:-1])
    y.append(0 if data[-1] == "<=50K" else 1)

X = np.array(X)
y = np.array(y)

X_encoded = np.empty(X.shape)

for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])

X = X_encoded.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

model = OneVsOneClassifier(LinearSVC())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== TASK 2.1: LINEAR SVM ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
