import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

X_encoded = []

for col in range(X.shape[1]):
    le = preprocessing.LabelEncoder()
    col_data = le.fit_transform(X[:, col])
    X_encoded.append(col_data)

X = np.array(X_encoded).T
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)


kernels = ["poly", "rbf", "sigmoid"]

print("=== TASK 2.2: KERNEL SVM ===")

for k in kernels:
    model = SVC(kernel=k, degree=3)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nKernel:", k)
    print("Accuracy:", accuracy_score(y_test, pred))
