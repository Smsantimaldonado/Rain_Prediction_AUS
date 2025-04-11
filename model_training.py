import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('weatherAUS_clean.csv')

sfs = SFS(LogisticRegression(), k_features=12, forward=True, floating=False, 
          scoring = 'accuracy', cv = 5)

X_lr = df.drop(['RainTomorrow'], axis=1)
y_lr = df['RainTomorrow']

sfs.fit(X_lr, y_lr)
variables_elegidas = list(sfs.k_feature_names_)

X_lr = df[variables_elegidas]
y_lr = df['raintomorrow']

X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size = 0.3, random_state = 123, shuffle=True, stratify=y_lr)

model_lr = LogisticRegression(max_iter=10000)
model_lr.fit(X_lr_train, y_lr_train)

y_lr_pred = model_lr.predict(X_lr_test)
print(classification_report(y_true=y_lr_test, y_pred=y_lr_pred))


sfs = SFS(DecisionTreeClassifier(), k_features=10, forward=False, scoring = 'accuracy', cv = 5)

X_dtc = df.drop(['raintomorrow', 'date', 'Longitud', 'Latitud', 'Location'], axis=1)
y_dtc = df['raintomorrow']

sfs.fit(X_dtc, y_dtc)
variables_elegidas = list(sfs.k_feature_names_)

X_dtc = df[variables_elegidas]
y_dtc = df['raintomorrow']

X_dtc_train, X_dtc_test, y_dtc_train, y_dtc_test = train_test_split(X_dtc, y_dtc, test_size = .3, random_state = 123, shuffle=True, stratify=y_dtc)

clf = DecisionTreeClassifier(random_state=123, criterion = 'entropy', max_depth=10)
model_dtc = clf.fit(X_dtc_train, y_dtc_train)

y_dtc_pred = clf.predict(X_dtc_test)
print(classification_report(y_true=y_dtc_test, y_pred=y_dtc_pred))


matriz = confusion_matrix(y_dtc_test, y_dtc_pred)
print(matriz)


y_dtc_prob = model_dtc.predict_proba(X_dtc_test)[:, 1]
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_dtc_test, y_dtc_prob)
roc_auc = auc(fpr, tpr)