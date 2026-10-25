import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def train_and_evaluate_models(
    data_path='data/ev_battery_degradation.csv', 
    artifacts_dir='artifacts'
):
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature columns
    cat_features = ['Car_Model', 'Battery_Type', 'Driving_Style']
    num_features = [
        'Battery_Capacity_kWh',
        'Vehicle_Age_Months',
        'Total_Charging_Cycles',
        'Avg_Temperature_C',
        'Fast_Charge_Ratio',
        'Avg_Discharge_Rate_C',
        'Internal_Resistance_Ohm'
    ]
    target_col = 'SoH_Percent'
    
    X = df[cat_features + num_features]
    y = df[target_col]
    
    # Reproducible 70% Train, 15% Validation, 15% Test Split
    # Step 1: 70% train, 30% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    # Step 2: split temp into 15% val and 15% test (50% of temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )
    
    print(f"Dataset split: Train={len(X_train)} (70%), Val={len(X_val)} (15%), Test={len(X_test)} (15%)")
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=12, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    }
    
    metrics = {}
    fitted_pipelines = {}
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit on training data ONLY to prevent data leakage
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        
        # Evaluate on Validation Set
        val_preds = pipeline.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_rmse = root_mean_squared_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)
        
        # Evaluate on Held-out Test Set
        test_preds = pipeline.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = root_mean_squared_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        metrics[name] = {
            'validation': {
                'mae': round(float(val_mae), 3),
                'rmse': round(float(val_rmse), 3),
                'r2': round(float(val_r2), 4)
            },
            'test': {
                'mae': round(float(test_mae), 3),
                'rmse': round(float(test_rmse), 3),
                'r2': round(float(test_r2), 4)
            }
        }
        
        print(f"{name} Results:")
        print(f"  Val  -> MAE: {val_mae:.3f}, RMSE: {val_rmse:.3f}, R2: {val_r2:.4f}")
        print(f"  Test -> MAE: {test_mae:.3f}, RMSE: {test_rmse:.3f}, R2: {test_r2:.4f}")
        
        # Save each pipeline artifact
        artifact_filename = name.lower().replace(' ', '_')
        joblib.dump(pipeline, os.path.join(artifacts_dir, f"model_{artifact_filename}.joblib"))
    
    # Identify Best Model by Validation R2
    best_model_name = max(metrics, key=lambda m: metrics[m]['validation']['r2'])
    metrics['best_model'] = best_model_name
    print(f"\nBest Model identified: {best_model_name}")
    
    # Save metrics JSON
    with open(os.path.join(artifacts_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Feature Importance (from Random Forest)
    rf_pipeline = fitted_pipelines['Random Forest']
    rf_regressor = rf_pipeline.named_steps['regressor']
    ohe_cols = list(rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features))
    all_feature_names = num_features + ohe_cols
    importances = rf_regressor.feature_importances_
    
    # Aggregate one-hot back to feature levels for cleaner visualization
    aggregated_importance = {}
    for feat in num_features:
        idx = all_feature_names.index(feat)
        aggregated_importance[feat] = float(importances[idx])
        
    for cat in cat_features:
        cat_total = sum(
            float(importances[i]) for i, name in enumerate(all_feature_names) if name.startswith(cat)
        )
        aggregated_importance[cat] = float(cat_total)
        
    sorted_importance = dict(sorted(aggregated_importance.items(), key=lambda item: item[1], reverse=True))
    with open(os.path.join(artifacts_dir, 'feature_importance.json'), 'w') as f:
        json.dump(sorted_importance, f, indent=2)
        
    # Scatter plot sample points for Actual vs Predicted (Random Forest test set, 150 sampled points)
    rf_test_preds = fitted_pipelines['Random Forest'].predict(X_test)
    sample_indices = np.random.choice(len(y_test), size=min(150, len(y_test)), replace=False)
    scatter_data = [
        {
            'actual': round(float(y_test.iloc[i]), 2),
            'predicted': round(float(rf_test_preds[i]), 2)
        }
        for i in sample_indices
    ]
    with open(os.path.join(artifacts_dir, 'scatter_sample.json'), 'w') as f:
        json.dump(scatter_data, f, indent=2)
        
    # Distribution sample for SoH Histogram
    counts, bin_edges = np.histogram(df[target_col], bins=20)
    distribution_data = {
        'bins': [f"{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}%" for i in range(len(counts))],
        'counts': [int(c) for c in counts]
    }
    with open(os.path.join(artifacts_dir, 'distribution_sample.json'), 'w') as f:
        json.dump(distribution_data, f, indent=2)
        
    print("\nAll ML models, preprocessing pipelines, and evaluation metrics successfully saved to artifacts/")

if __name__ == '__main__':
    train_and_evaluate_models()
