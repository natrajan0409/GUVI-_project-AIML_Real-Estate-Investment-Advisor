"""
Model Training Module for Real Estate Investment Advisor
Trains classification and regression models with MLflow tracking
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            mean_squared_error, mean_absolute_error, r2_score)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')


class RealEstateModelTrainer:
    def __init__(self, data_path):
        """Initialize model trainer"""
        self.df = pd.read_csv(data_path)
        self.X_train_clf = None
        self.X_test_clf = None
        self.y_train_clf = None
        self.y_test_clf = None
        self.X_train_reg = None
        self.X_test_reg = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.scaler = StandardScaler()
        
        print(f"Data loaded for model training: {self.df.shape}")
        
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\n" + "="*80)
        print("PREPARING FEATURES FOR MODELING")
        print("="*80)
        
        # Select features (use encoded columns)
        feature_cols = [
            'BHK', 'Size_in_SqFt', 'Price_per_SqFt', 'Year_Built',
            'Floor_No', 'Total_Floors', 'Age_of_Property',
            'Nearby_Schools', 'Nearby_Hospitals',
            'State_Encoded', 'City_Encoded', 'Locality_Encoded',
            'Property_Type_Encoded', 'Furnished_Status_Encoded',
            'Facing_Encoded', 'Owner_Type_Encoded', 
            'Availability_Status_Encoded', 'Security_Encoded'
        ]
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        # For classification (exclude price-related features to avoid data leakage)
        clf_features = [col for col in feature_cols if 'Price' not in col]
        
        # For regression (include current price)
        reg_features = feature_cols.copy()
        if 'Price_in_Lakhs' not in reg_features:
            reg_features.append('Price_in_Lakhs')
        
        print(f"\nClassification features: {len(clf_features)}")
        print(f"Regression features: {len(reg_features)}")
        
        return clf_features, reg_features
    
    def split_data_classification(self, clf_features):
        """Split data for classification"""
        X = self.df[clf_features]
        y = self.df['Good_Investment']
        
        self.X_train_clf, self.X_test_clf, self.y_train_clf, self.y_test_clf = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nClassification split:")
        print(f"  Training set: {self.X_train_clf.shape}")
        print(f"  Test set: {self.X_test_clf.shape}")
        
    def split_data_regression(self, reg_features):
        """Split data for regression"""
        X = self.df[reg_features]
        y = self.df['Future_Price_5Y']
        
        self.X_train_reg, self.X_test_reg, self.y_train_reg, self.y_test_reg = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nRegression split:")
        print(f"  Training set: {self.X_train_reg.shape}")
        print(f"  Test set: {self.X_test_reg.shape}")
    
    def cleanup_old_experiments(self):
        """Delete old experiment runs to start fresh"""
        print("\n" + "="*80)
        print("CLEANING UP OLD MLFLOW EXPERIMENTS")
        print("="*80)
        
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Get experiments
            experiments = ['Real_Estate_Classification', 'Real_Estate_Regression']
            
            for exp_name in experiments:
                try:
                    experiment = client.get_experiment_by_name(exp_name)
                    if experiment:
                        # Get all runs for this experiment
                        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
                        
                        if runs:
                            print(f"\nDeleting {len(runs)} old runs from '{exp_name}'...")
                            for run in runs:
                                client.delete_run(run.info.run_id)
                            print(f"  [+] Cleared {len(runs)} runs")
                        else:
                            print(f"\n'{exp_name}': No previous runs found")
                except Exception as e:
                    print(f"\n'{exp_name}': No previous experiment found (first run)")
            
            print("\n[+] Cleanup complete! Starting fresh training...")
            
        except Exception as e:
            print(f"\nNote: Could not clean old experiments (this is OK for first run)")
            print(f"  Reason: {str(e)}")

    
    def train_classification_models(self):
        """Train multiple classification models with MLflow tracking"""
        print("\n" + "="*80)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*80)
        
        mlflow.set_experiment("Real_Estate_Classification")
        
        models = {
            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            'Decision_Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Naive_Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*40}")
            print(f"Training: {model_name}")
            print(f"{'='*40}")
            
            with mlflow.start_run(run_name=model_name):
                # Train model
                model.fit(self.X_train_clf, self.y_train_clf)
                
                # Predictions
                y_pred = model.predict(self.X_test_clf)
                y_pred_proba = model.predict_proba(self.X_test_clf)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Metrics
                accuracy = accuracy_score(self.y_test_clf, y_pred)
                precision = precision_score(self.y_test_clf, y_pred)
                recall = recall_score(self.y_test_clf, y_pred)
                f1 = f1_score(self.y_test_clf, y_pred)
                roc_auc = roc_auc_score(self.y_test_clf, y_pred_proba) if y_pred_proba is not None else 0
                
                # Log parameters
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("test_size", 0.2)
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)
                
                # Log model
                mlflow.sklearn.log_model(model, name=model_name)
                
                # Store results
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'model': model
                }
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                print(f"ROC AUC: {roc_auc:.4f}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        print(f"\n{'='*80}")
        print(f"BEST CLASSIFICATION MODEL: {best_model_name}")
        print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
        print(f"{'='*80}")
        
        return results, best_model_name
    
    def train_regression_models(self):
        """Train multiple regression models with MLflow tracking"""
        print("\n" + "="*80)
        print("TRAINING REGRESSION MODELS")
        print("="*80)
        
        mlflow.set_experiment("Real_Estate_Regression")
        
        models = {
            'Linear_Regression': LinearRegression(),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'Decision_Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*40}")
            print(f"Training: {model_name}")
            print(f"{'='*40}")
            
            with mlflow.start_run(run_name=model_name):
                # Train model
                model.fit(self.X_train_reg, self.y_train_reg)
                
                # Predictions
                y_pred = model.predict(self.X_test_reg)
                
                # Metrics
                rmse = np.sqrt(mean_squared_error(self.y_test_reg, y_pred))
                mae = mean_absolute_error(self.y_test_reg, y_pred)
                r2 = r2_score(self.y_test_reg, y_pred)
                
                # Log parameters
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("test_size", 0.2)
                
                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2_score", r2)
                
                # Log model
                mlflow.sklearn.log_model(model, name=model_name)
                
                # Store results
                results[model_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'model': model
                }
                
                print(f"RMSE: {rmse:.4f}")
                print(f"MAE: {mae:.4f}")
                print(f"R2 Score: {r2:.4f}")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['r2_score'])
        print(f"\n{'='*80}")
        print(f"BEST REGRESSION MODEL: {best_model_name}")
        print(f"R2 Score: {results[best_model_name]['r2_score']:.4f}")
        print(f"{'='*80}")
        
        return results, best_model_name
    
    def run_full_training(self):
        """Run complete training pipeline"""
        print("\n" + "="*80)
        print("STARTING MODEL TRAINING PIPELINE")
        print("="*80)
        
        # Clean up old experiments first
        self.cleanup_old_experiments()
        
        # Prepare features
        clf_features, reg_features = self.prepare_features()
        
        # Split data
        self.split_data_classification(clf_features)
        self.split_data_regression(reg_features)
        
        # Train classification models
        clf_results, best_clf = self.train_classification_models()
        
        # Train regression models
        reg_results, best_reg = self.train_regression_models()
        
        print("\n" + "="*80)
        print("[SUCCESS] MODEL TRAINING COMPLETED!")
        print("="*80)
        print(f"\nBest Classification Model: {best_clf}")
        print(f"Best Regression Model: {best_reg}")
        print("\nView results in MLflow UI: mlflow ui")
        
        return clf_results, reg_results, best_clf, best_reg


def main():
    """Main function to run model training"""
    data_path = "data/processed/housing_data_processed.csv"
    
    trainer = RealEstateModelTrainer(data_path)
    clf_results, reg_results, best_clf, best_reg = trainer.run_full_training()
    
    print("\n[SUCCESS] Training complete! Run 'mlflow ui' to view experiments.")


if __name__ == "__main__":
    main()
