import joblib
import os
from datetime import datetime


results = joblib.load('data/model_output/training_results.pkl')

# Create exit folder if does not exist
output_dir = 'data/model_output'
os.makedirs(output_dir, exist_ok=True)

# Pick best model by AUC
best_model_name = max(results, key=lambda x: results[x]['auc'] if results[x]['auc'] != 'N/A' else -1)
best_model = results[best_model_name]['model']
selected_features = results[best_model_name]['features']
le = results[best_model_name]['encoder']

# Save model, features and label encoder
joblib.dump(best_model, f'{output_dir}/best_model_{best_model_name}.pkl')
joblib.dump(selected_features, f'{output_dir}/selected_features.pkl')
joblib.dump(le, f'{output_dir}/label_encoder.pkl')

# Log
print(f"\nModel '{best_model_name}' saved with AUC: {results[best_model_name]['auc']:.4f}")
print(f'\nModel location: {output_dir}/best_model_{best_model_name}.pkl')
print(f'\nFeatures location: {output_dir}/selected_features.pkl')
print(f'\nEncoder location: {output_dir}/label_encoder.pkl')
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f'\n✅ Process completed successfully at {now}')