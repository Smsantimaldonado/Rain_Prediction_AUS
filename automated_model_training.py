import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from feature_eng_pipeline import DataPreprocessor


# Models to compare
models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10, criterion='entropy'),
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}

# Load raw data
df = pd.read_csv('weatherAUS.csv')
df = df.drop(columns=['RISK_MM'], errors='ignore')

# Apply preproccessing
preprocessor = DataPreprocessor()
df_clean = preprocessor.fit_transform(df)

# Split X, y
X = df_clean.drop(columns='RainTomorrow')
y = df_clean['RainTomorrow']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

# To save results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f'\n🧠 Training model: {name}\n')

    # Feature selection
    sfs = SFS(model,
              k_features=10,
              forward=True,
              floating=False,
              scoring='accuracy',
              cv=5,
              n_jobs=-1)

    print('Starting to select features')
    sfs.fit(X_train, y_train)

    if sfs.k_feature_names_ is not None:
        selected_features = list(sfs.k_feature_names_)
        print(f'\n🧠 Selected features: {selected_features}')
    else:
        raise ValueError(f"SFS did not select any feature for model {name}. Review input data.")

    # Train with best features
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    model.fit(X_train_sel, y_train)

    y_pred = model.predict(X_test_sel)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_sel)[:, 1]
    else:
        y_prob = None

    print('\n📊 Classification Report:')
    print(classification_report(y_test, y_pred))

    print('\n🧮 Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 'N/A'
    print(f'🔥 ROC AUC: {auc}')

    print('-'*30)

    results[name] = {
        'features': selected_features,
        'model': model,
        'auc': auc
    }

# Show final results
print('\n✅ Resumen de modelos:')
for name, res in results.items():
    print(f'{name}: {len(res['features'])} features - AUC: {res['auc']}')