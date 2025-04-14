# Rain_Prediction_AUS
Goal: Determine if it will rain the next day, in different cities in Australia, without relying solely on the repetition of values in the variables.

# Rain_Prediction_AUS
Goal: Determine if it will rain the next day, in different cities in Australia, without relying solely on the repetition of values in the variables.

graph TD
    A[📂 weatherAUS.csv] --> B[🧹 Preprocesamiento <br> DataPreprocessor()]
    B --> C[🧪 Entrenamiento de modelos <br> train_models_with_sfs()]
    C --> D[📦 Guardado de resultados <br> training_results.pkl + log.txt]
    D --> E[🏅 save_best_model.py <br> Selección del mejor modelo]
    E --> F[💾 pipeline_final.pkl <br> (preproc + modelo)]

    F --> G[🧠 predict_new_data.py <br> Predicción con datos nuevos]


🧠 Descripción de cada etapa
A. Dataset crudo:	weatherAUS.csv ubicado en data/raw/
B. Preprocesamiento:	Imputación, OHE, derivadas como TempRange, Month, reducción de direcciones de viento, etc.
C. Entrenamiento:	Se prueban 4 modelos con SequentialFeatureSelector para elegir las mejores features
D. Guardado de resultados:	Se guarda results.pkl y un training_log.txt con fecha, AUC y features
E. Selección del mejor modelo:	save_best_model.py elige el modelo con mejor ROC AUC y guarda pipeline_final.pkl
F. Pipeline completo:	Incluye preprocesamiento + modelo entrenado
G. Predicción en producción:	Se puede usar en un script, API o webapp: predict_new_data.py

📂 Archivos involucrados
Archivo	Rol
feature_eng_pipeline.py	Preprocesamiento de datos
model_training.py	Entrena, evalúa y guarda results.pkl
save_best_model.py	Selecciona y guarda el pipeline final
predict_new_data.py	Carga el pipeline entrenado y predice