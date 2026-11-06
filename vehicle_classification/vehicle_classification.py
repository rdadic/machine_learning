import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

from ucimlrepo import fetch_ucirepo


#vehicle silhouettes dataset
vehicle = fetch_ucirepo(id = 149)

X = vehicle.data.features
print(X.columns)
y = vehicle.data.targets
print(y)

df = X.copy()
df['target'] = y

print(df.head())
print(df.shape)
print(df['target'].value_counts())

# class: bus, van, opel, saab and one invalid value '204'

print(y['class'].unique())
print(y[y['class'] == '204'])

# identify and remove the invalid value
print('Index 752:', df.iloc[752])
print(df['target'].value_counts())
print(df.iloc[752].name)
print(y.iloc[752])

filter = y['class'] != '204'

X = X[filter].copy()
y = y[filter].copy()

print(X.shape)
print(y.shape)
print(y['class'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size = 0.2, 
    random_state = 42, 
    stratify = y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = LogisticRegression(max_iter = 2000)
classifier.fit(X_train_scaled, y_train.values.ravel())

y_predicted_lr = classifier.predict(X_test_scaled)

cm_lr = confusion_matrix(y_test, y_predicted_lr)

display_cm_lr = ConfusionMatrixDisplay(
    confusion_matrix = cm_lr,
    display_labels = classifier.classes_
)

display_cm_lr.plot(cmap = 'Blues')
plt.show()

print('Logistic regression - metrics: ')
print('Accuracy: ', accuracy_score(y_test, y_predicted_lr))
print('Precision: ', precision_score(y_test, y_predicted_lr, average='macro'))
print('Recall: ', recall_score(y_test, y_predicted_lr, average='macro'))
print('F1: ', f1_score(y_test, y_predicted_lr, average='macro'))
print('Balanced accuracy: ', balanced_accuracy_score(y_test, y_predicted_lr))

print(classification_report(y_test, y_predicted_lr))

rf_classifier = RandomForestClassifier(
    n_estimators = 100, 
    random_state = 42
)

rf_classifier.fit(X_train, y_train.values.ravel())

y_predicted_rf = rf_classifier.predict(X_test)

cm_rf = confusion_matrix(y_test, y_predicted_rf)

display_cm = ConfusionMatrixDisplay(
    confusion_matrix = cm_rf,
    display_labels = rf_classifier.classes_
)

display_cm.plot(cmap = 'Blues')

plt.show()

print('Random forest classification - metrics: ')
print(classification_report(y_test, y_predicted_rf))