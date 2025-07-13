import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyHitPredictor:
    """
    Machine learning models for Spotify hit prediction
    """
    
    def __init__(self, task_type: str = 'regression'):
        """
        Initialize the predictor
        
        Args:
            task_type: 'regression' for popularity score prediction or 'classification' for tier prediction
        """
        self.task_type = task_type
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
        self.model_scores = {}
        
    def get_models(self) -> Dict[str, Any]:
        """
        Get model dictionary based on task type
        
        Returns:
            Dictionary of model names and model objects
        """
        if self.task_type == 'regression':
            return {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'svr': SVR(kernel='rbf'),
                'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
        else:  # classification
            return {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'svc': SVC(kernel='rbf', random_state=42),
                'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Train/test split data
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if self.task_type == 'classification' else None
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to maintain column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        logger.info(f"Data prepared: {X_train_scaled.shape[0]} training samples, {X_test_scaled.shape[0]} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """
        Train all models and evaluate with cross-validation
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of model scores
        """
        models = self.get_models()
        results = {}
        
        logger.info(f"Training {len(models)} models with {cv_folds}-fold cross-validation...")
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Cross-validation
                if self.task_type == 'regression':
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                    metric_name = 'R2'
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                    metric_name = 'Accuracy'
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                results[name] = {
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'cv_scores': cv_scores
                }
                
                # Train the full model
                model.fit(X_train, y_train)
                self.models[name] = model
                
                logger.info(f"{name} - {metric_name}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        self.model_scores = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['mean_score'])
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model: {best_model_name}")
        
        return results
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test set
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                predictions = model.predict(X_test)
                
                if self.task_type == 'regression':
                    mse = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, predictions)
                    
                    results[name] = {
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'predictions': predictions
                    }
                    
                    logger.info(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
                    
                else:  # classification
                    accuracy = accuracy_score(y_test, predictions)
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'predictions': predictions,
                        'classification_report': classification_report(y_test, predictions, output_dict=True)
                    }
                    
                    logger.info(f"{name} - Accuracy: {accuracy:.4f}")
                    
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                continue
        
        return results
    
    def get_feature_importance(self, X: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from tree-based models
        
        Args:
            X: Feature DataFrame
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        importance_data = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_,
                    'model': name
                }).sort_values('importance', ascending=False)
                
                importance_data.append(importance)
        
        if importance_data:
            combined_importance = pd.concat(importance_data)
            
            # Average importance across models
            avg_importance = combined_importance.groupby('feature')['importance'].mean().reset_index()
            avg_importance = avg_importance.sort_values('importance', ascending=False).head(top_n)
            
            self.feature_importance = avg_importance
            return avg_importance
        
        return pd.DataFrame()
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, model_name: str = 'random_forest') -> Dict:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of model to tune
            
        Returns:
            Best parameters and score
        """
        logger.info(f"Hyperparameter tuning for {model_name}...")
        
        if self.task_type == 'regression':
            if model_name == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                model = RandomForestRegressor(random_state=42)
                scoring = 'r2'
                
            elif model_name == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
                model = GradientBoostingRegressor(random_state=42)
                scoring = 'r2'
                
        else:  # classification
            if model_name == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                model = RandomForestClassifier(random_state=42)
                scoring = 'accuracy'
                
            elif model_name == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
                model = GradientBoostingClassifier(random_state=42)
                scoring = 'accuracy'
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring=scoring, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best params
        self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    def save_models(self, save_dir: str = 'saved_models'):
        """
        Save trained models to disk
        
        Args:
            save_dir: Directory to save models
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for name, model in self.models.items():
            filepath = os.path.join(save_dir, f'{name}_{self.task_type}.joblib')
            joblib.dump(model, filepath)
            logger.info(f"Saved {name} to {filepath}")
        
        # Save scaler
        scaler_path = os.path.join(save_dir, f'scaler_{self.task_type}.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = os.path.join(save_dir, f'feature_importance_{self.task_type}.csv')
            self.feature_importance.to_csv(importance_path, index=False)
    
    def load_models(self, save_dir: str = 'saved_models'):
        """
        Load trained models from disk
        
        Args:
            save_dir: Directory containing saved models
        """
        model_files = [f for f in os.listdir(save_dir) if f.endswith(f'{self.task_type}.joblib') and 'scaler' not in f]
        
        for file in model_files:
            model_name = file.replace(f'_{self.task_type}.joblib', '')
            filepath = os.path.join(save_dir, file)
            self.models[model_name] = joblib.load(filepath)
            logger.info(f"Loaded {model_name} from {filepath}")
        
        # Load scaler
        scaler_path = os.path.join(save_dir, f'scaler_{self.task_type}.joblib')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
    
    def predict_single_track(self, track_features: Dict[str, float]) -> float:
        """
        Predict popularity for a single track
        
        Args:
            track_features: Dictionary of track features
            
        Returns:
            Predicted popularity score or class
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train models first.")
        
        # Convert to DataFrame
        features_df = pd.DataFrame([track_features])
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)[0]
        
        return prediction

class ModelEvaluator:
    """
    Class to evaluate and compare models
    """
    
    def __init__(self):
        self.results = {}
    
    def plot_model_comparison(self, model_scores: Dict, save_path: str = None):
        """
        Plot model comparison
        
        Args:
            model_scores: Dictionary of model scores
            save_path: Path to save plot
        """
        models = list(model_scores.keys())
        scores = [model_scores[model]['mean_score'] for model in models]
        errors = [model_scores[model]['std_score'] for model in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, scores, yerr=errors, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors)*0.1, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, save_path: str = None, top_n: int = 15):
        """
        Plot feature importance
        
        Args:
            feature_importance: DataFrame with feature importance
            save_path: Path to save plot
            top_n: Number of top features to show
        """
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str = None):
        """
        Plot predictions vs actual values (for regression)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Popularity')
        plt.ylabel('Predicted Popularity')
        plt.title(f'Predictions vs Actual - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R2 score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function for model training and evaluation"""
    import sys
    import os
    sys.path.append('../DataCollection')
    from config import DATA_DIR
    
    # Load engineered features
    data_path = os.path.join('..', DATA_DIR, 'engineered_features.csv')
    
    if not os.path.exists(data_path):
        print(f"Engineered features not found at {data_path}")
        print("Please run data collection and feature engineering first.")
        return
    
    df = pd.read_csv(data_path)
    
    # Prepare features for modeling
    sys.path.append('../DataCollection')
    from feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    
    print("="*50)
    print("SPOTIFY HIT PREDICTOR - MODEL TRAINING")
    print("="*50)
    
    # Regression task
    print("\n1. REGRESSION TASK (Predict exact popularity score)")
    print("-" * 50)
    
    X_reg, y_reg = engineer.prepare_for_modeling(df, target_column='popularity')
    
    # Initialize regression predictor
    reg_predictor = SpotifyHitPredictor(task_type='regression')
    
    # Prepare data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = reg_predictor.prepare_data(X_reg, y_reg)
    
    # Train models
    reg_scores = reg_predictor.train_models(X_train_reg, y_train_reg)
    
    # Evaluate models
    reg_results = reg_predictor.evaluate_models(X_test_reg, y_test_reg)
    
    # Feature importance
    reg_importance = reg_predictor.get_feature_importance(X_reg)
    
    # Classification task
    print("\n2. CLASSIFICATION TASK (Predict popularity tier)")
    print("-" * 50)
    
    X_clf, y_clf = engineer.prepare_for_modeling(df, target_column='popularity_tier')
    
    # Remove rows with NaN target (if any)
    mask = ~y_clf.isnull()
    X_clf = X_clf[mask]
    y_clf = y_clf[mask]
    
    # Initialize classification predictor
    clf_predictor = SpotifyHitPredictor(task_type='classification')
    
    # Prepare data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = clf_predictor.prepare_data(X_clf, y_clf)
    
    # Train models
    clf_scores = clf_predictor.train_models(X_train_clf, y_train_clf)
    
    # Evaluate models
    clf_results = clf_predictor.evaluate_models(X_test_clf, y_test_clf)
    
    # Feature importance
    clf_importance = clf_predictor.get_feature_importance(X_clf)
    
    # Save models
    reg_predictor.save_models('saved_models')
    clf_predictor.save_models('saved_models')
    
    # Visualization
    evaluator = ModelEvaluator()
    
    print("\n3. RESULTS SUMMARY")
    print("-" * 50)
    
    print("\nRegression Results:")
    for model, results in reg_results.items():
        print(f"{model}: RMSE = {results['rmse']:.3f}, R² = {results['r2']:.3f}")
    
    print("\nClassification Results:")
    for model, results in clf_results.items():
        print(f"{model}: Accuracy = {results['accuracy']:.3f}")
    
    if reg_importance is not None and not reg_importance.empty:
        print("\nTop 10 Important Features (Regression):")
        print(reg_importance.head(10)[['feature', 'importance']].to_string(index=False))
    
    # Create visualizations
    evaluator.plot_model_comparison(reg_scores, 'plots/regression_comparison.png')
    evaluator.plot_model_comparison(clf_scores, 'plots/classification_comparison.png')
    
    if reg_importance is not None and not reg_importance.empty:
        evaluator.plot_feature_importance(reg_importance, 'plots/feature_importance_regression.png')
    
    if clf_importance is not None and not clf_importance.empty:
        evaluator.plot_feature_importance(clf_importance, 'plots/feature_importance_classification.png')
    
    # Plot predictions vs actual for best regression model
    best_reg_model = max(reg_results.keys(), key=lambda x: reg_results[x]['r2'])
    evaluator.plot_predictions_vs_actual(
        y_test_reg, reg_results[best_reg_model]['predictions'], 
        best_reg_model, 'plots/predictions_vs_actual.png'
    )
    
    print(f"\nBest regression model: {best_reg_model}")
    print(f"Best classification model: {max(clf_results.keys(), key=lambda x: clf_results[x]['accuracy'])}")
    
    print("\nModels saved to 'saved_models' directory")
    print("Plots saved to 'plots' directory")

if __name__ == "__main__":
    main() 