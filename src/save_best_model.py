import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from feature_eng_pipeline import DataPreprocessor
from automated_model_training import results


# Create exit folder if does not exist
output_dir = "model_output"
os.makedirs(output_dir, exist_ok=True)

# Pick best model by AUC
best_model_name = max(results, key=lambda x: results[x]['auc'] if results[x]['auc'] != "N/A" else -1)
best_model = results[best_model_name]['model']
best_features = results[best_model_name]['features']

# Save full pipeline for preproccesing + model
final_pipeline = Pipeline([
    ('preprocessing', DataPreprocessor()),
    ('feature_selection', 'passthrough'),  # placeholder
    ('model', best_model)
])

# ⚠️ Warning:
# It is not possible to include SFS directly in the pipeline because it depends on the model
# Solution: saving the model trained with the selected features

# Save model, features and label encoder
le = LabelEncoder()
joblib.dump(best_model, f"{output_dir}/best_model_{best_model_name}.pkl")
joblib.dump(best_features, f"{output_dir}/best_features.pkl")
joblib.dump(le, f"{output_dir}/label_encoder.pkl")

print(f"\n💾 Modelo '{best_model_name}' guardado con AUC: {results[best_model_name]['auc']:.4f}")
print(f"📁 Ubicación: {output_dir}/best_model_{best_model_name}.pkl")