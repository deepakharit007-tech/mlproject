import os
import sys
from dataclasses import dataclass

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Decision Tree': DecisionTreeRegressor(),
                'KNN': KNeighborsRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Linear Regression': LinearRegression()
            }

            if XGBRegressor is not None:
                models['XGBoost'] = XGBRegressor()

            if CatBoostRegressor is not None:
                models['CatBoost'] = CatBoostRegressor()


            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            if not model_report:
                raise CustomException('No models were evaluated', sys)

            # Select best model by score (deterministic)
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found', sys)

            logging.info(f'Best model found: {best_model_name} with score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return {
                'model_name': best_model_name,
                'model_score': best_model_score,
                'r2_score': r2_square,
                'model': best_model
            }
        except Exception as e:
            raise CustomException(e, sys)

