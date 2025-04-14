import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from datetime import datetime


def train_models_with_sfs(df, models_dict, preprocessor, target_column='RainTomorrow'):
    """
    Preprocesses the data, trains each model using SFS, evaluates them, and returns the results.

    Parameters:
    - df: raw input DataFrame
    - models_dict: dictionary with model names and sklearn model instances
    - preprocessor: a fitted or unfitted DataPreprocessor instance
    - target_column: name of the column to use as the target variable (default: 'RainTomorrow')

    Returns:
    - results: dictionary with model, selected features and AUC for each trained model
    """
    print("🔄 Applying preprocessing...")

    # Apply preproccessing
    df_clean = preprocessor.fit_transform(df)

    # Split X, y
    X = df_clean.drop(columns=target_column)
    y = df_clean[target_column]

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

    # To save results
    results = {}

    # Train and evaluate each model
    for name, model in models_dict.items():
        print(f'\n🧠 Training model: {name}')

        # Feature selection
        sfs = SFS(model,
                    k_features=10,
                    forward=True,
                    floating=False,
                    scoring='accuracy',
                    cv=5,
                    n_jobs=-1)

        print('\nStarting feature selection')
        sfs.fit(X_train, y_train)

        if sfs.k_feature_names_ is not None:
            selected_features = list(sfs.k_feature_names_)
            print(f'\n🧠 Selected features: {selected_features}')
        elif sfs.k_feature_names_ is None:
            raise ValueError(f"⚠️ SFS failed for model {name}. Review input data.")
        
        # Train model with selected features
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
            'model': model,
            'features': selected_features,
            'auc': auc,
            'encoder': le
        }

    return results


# ===========================
# MAIN BLOCK FOR EXECUTION
# ===========================
if __name__ == '__main__':
    from feature_eng_pipeline import DataPreprocessor
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import joblib

    # Models to compare
    models = {
        'LogisticRegression': LogisticRegression(max_iter=10000),
        'DecisionTree': DecisionTreeClassifier(max_depth=10),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }

    # Load raw data
    df = pd.read_csv('data/raw/weatherAUS.csv')
    df.drop(columns='RISK_MM', errors='ignore', inplace=True)

    # Create location to save final results
    output_dir = 'data/model_output'
    os.makedirs(output_dir, exist_ok=True)

    results = None

    try:
        # Training
        results = train_models_with_sfs(df, models, preprocessor=DataPreprocessor())
    finally:
        if results is not None:
            joblib.dump(results, os.path.join(output_dir, 'training_results.pkl'))
            print("\n✅ Results saved in: data/model_output/training_results.pkl")
            log_path = os.path.join(output_dir, 'training_log.txt')
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\nTraining time: {now}\n")
                f.write("="*50 + "\n")
                for model_name, info in results.items():
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Features: {info['features']}\n")
                    f.write(f"ROC AUC: {round(info['auc'], 4)}\n\n")
            print(f"📝 Results log saved in: {log_path}")
        else:
            print("⚠️ Results were not saved because training failed.")