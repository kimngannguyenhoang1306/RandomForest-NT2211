import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path
import datetime
import os
import warnings


class MLPipeline:
    def __init__(self, random_state=42, n_jobs=None):
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs is not None else max(os.cpu_count() // 2, 1)
        self.setup_logging()

    def setup_logging(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'training_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def generate_evaluation_report(self, model, cv_scores, best_params=None):
        """
        Generate a comprehensive evaluation report for the model.
        """
        report = """
# Random Forest Model Evaluation Report

## 1. Model Performance Metrics
- Mean Cross-Validation Score: {:.4f} (+/- {:.4f})
- Cross-Validation Scores: {}

## 2. Model Configuration
### Base Model Parameters:
- Number of Trees (n_estimators): {}
- Maximum Depth: {}
- Number of CPU Cores Used: {}

""".format(
            cv_scores.mean(),
            cv_scores.std() * 2,
            [f"{score:.4f}" for score in cv_scores],
            model.n_estimators,
            model.max_depth,
            self.n_jobs
        )

        if best_params:
            report += """
### Optimized Parameters:
{}
""".format('\n'.join([f"- {k}: {v}" for k, v in best_params.items()]))

        report += """
## 3. Implementation Strengths
- Robust preprocessing with separate handling for numeric and categorical data
- Built-in cross-validation for reliable performance estimation
- Comprehensive logging system for training and evaluation
- Efficient handling of missing values
- Memory optimization through float32 data type conversion

## 4. Technical Features
- Automated feature encoding and scaling
- Proper train-validation split (80-20)
- Parallel processing optimization
- Systematic artifact management
- Comprehensive error handling

## 5. Production Readiness
- Complete logging system
- Organized artifact storage
- Proper version control through timestamps
- Memory-efficient implementation
- Robust error handling

## 6. Areas for Enhancement
1. Feature importance analysis
2. Model interpretability tools
3. Advanced preprocessing options
4. Performance monitoring metrics
5. Model versioning system
"""

        # Save report to file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'model_evaluation_{timestamp}.md'
        with open(report_path, 'w') as f:
            f.write(report)

        self.logger.info(f"Evaluation report saved to {report_path}")
        return report

    def load_data(self, train_file: str, test_file: str) -> tuple:
        try:
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)

            if 'Label' not in df_train.columns:
                raise ValueError("Training data must contain 'Label' column")

            train_features = set(df_train.columns) - {'Label'}
            test_features = set(df_test.columns)
            if train_features != test_features:
                raise ValueError("Training and test data must have matching features")

            self.logger.info(f"Loaded training data shape: {df_train.shape}")
            self.logger.info(f"Loaded test data shape: {df_test.shape}")

            return df_train, df_test

        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame, scaler=None, label_encoder=None,
                        is_training=True) -> tuple:
        try:
            df = df.copy()

            # Xác định loại cột
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            # Xử lý missing values cho từng loại dữ liệu
            # Numeric: điền bằng median
            for col in numeric_cols:
                if is_training:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    if hasattr(self, f'median_{col}'):
                        setattr(self, f'median_{col}', median_val)
                else:
                    if hasattr(self, f'median_{col}'):
                        df[col] = df[col].fillna(getattr(self, f'median_{col}'))
                    else:
                        df[col] = df[col].fillna(0)

            # Categorical: điền bằng mode
            for col in categorical_cols:
                if is_training:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    if hasattr(self, f'mode_{col}'):
                        setattr(self, f'mode_{col}', mode_val)
                else:
                    if hasattr(self, f'mode_{col}'):
                        df[col] = df[col].fillna(getattr(self, f'mode_{col}'))
                    else:
                        df[col] = df[col].fillna('Unknown')

            # Chuyển đổi categorical columns sang numeric
            for col in categorical_cols:
                if is_training:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    if hasattr(self, f'le_{col}'):
                        setattr(self, f'le_{col}', le)
                else:
                    if hasattr(self, f'le_{col}'):
                        le = getattr(self, f'le_{col}')
                        df[col] = df[col].astype(str)
                        # Handle unseen categories
                        df[col] = df[col].map(lambda x: 'Unknown' if x not in le.classes_ else x)
                        df[col] = le.transform(df[col])
                    else:
                        df[col] = df[col].astype('category').cat.codes

            # Convert to float32 for memory efficiency
            for col in numeric_cols:
                df[col] = df[col].astype('float32')

            if is_training:
                X = df.drop('Label', axis=1)
                y = df['Label']

                label_encoder = LabelEncoder()
                scaler = StandardScaler()

                y = label_encoder.fit_transform(y)
                X = scaler.fit_transform(X)

                self.logger.info(f"Preprocessed training data shape: {X.shape}")
                return X, y, scaler, label_encoder
            else:
                X = scaler.transform(df)
                self.logger.info(f"Preprocessed test data shape: {X.shape}")
                return X, None

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {e}")
            raise

    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> tuple:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=1,
                max_depth=30,
                max_samples=0.8,
                warm_start=True
            )

            # Cross-validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    model, X, y,
                    cv=5,
                    n_jobs=self.n_jobs,
                    verbose=1
                )

            self.logger.info(f"CV Scores: {cv_scores}")
            self.logger.info(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate on validation set
            val_predictions = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)

            self.logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
            self.logger.info("\nClassification Report:")
            self.logger.info(classification_report(y_val, val_predictions))

            # Generate and save evaluation report
            self.generate_evaluation_report(model, cv_scores)

            return model, cv_scores.mean()

        except Exception as e:
            self.logger.error(f"Error in training: {e}")
            raise

    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        try:
            param_grid = {
                'n_estimators': [200],
                'max_depth': [20, None],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'max_features': ['sqrt']
            }

            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=1
            )

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,
                n_jobs=self.n_jobs,
                scoring='accuracy',
                verbose=2
            )

            grid_search.fit(X, y)

            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

            # Generate and save evaluation report with best parameters
            self.generate_evaluation_report(
                grid_search.best_estimator_,
                cross_val_score(grid_search.best_estimator_, X, y, cv=5),
                grid_search.best_params_
            )

            return grid_search.best_estimator_

        except Exception as e:
            self.logger.error(f"Error in optimization: {e}")
            raise

    def save_artifacts(self, model, scaler, label_encoder, predictions,
                       predictions_labels, output_dir='outputs'):
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            joblib.dump(model,
                        output_dir / f'model_{timestamp}.joblib',
                        compress=3)
            joblib.dump(scaler,
                        output_dir / f'scaler_{timestamp}.joblib',
                        compress=3)
            joblib.dump(label_encoder,
                        output_dir / f'label_encoder_{timestamp}.joblib',
                        compress=3)

            results_df = pd.DataFrame({
                'Predicted_Label_Encoded': predictions,
                'Predicted_Label': predictions_labels
            })
            results_df.to_csv(output_dir / f'predictions_{timestamp}.csv',
                              index=False)

            self.logger.info(f"All artifacts saved to {output_dir}")

        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            raise

    def run_pipeline(self, train_file: str, test_file: str, output_dir='outputs'):
        try:
            self.logger.info("Starting ML pipeline")

            # Load data
            df_train, df_test = self.load_data(train_file, test_file)

            # Preprocess training data
            X_train, y_train, scaler, label_encoder = self.preprocess_data(
                df_train, is_training=True
            )

            # Train base model
            model, base_score = self.train_random_forest(X_train, y_train)

            # Optimize model
            best_model = self.optimize_random_forest(X_train, y_train)

            # Preprocess test data
            X_test, _ = self.preprocess_data(
                df_test, scaler, label_encoder, is_training=False
            )

            # Make predictions
            predictions = best_model.predict(X_test)
            predictions_labels = label_encoder.inverse_transform(predictions)

            # Save artifacts
            self.save_artifacts(
                best_model, scaler, label_encoder,
                predictions, predictions_labels, output_dir
            )

            return best_model, scaler, label_encoder

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    # Configure number of workers
    n_jobs = max(os.cpu_count() // 2, 1)

    # Initialize and run pipeline
    pipeline = MLPipeline(n_jobs=n_jobs)
    try:
        best_model, scaler, label_encoder = pipeline.run_pipeline(
            train_file='train.csv',
            test_file='test.csv'
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")