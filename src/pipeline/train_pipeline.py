from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_object, load_object
from sklearn.pipeline import Pipeline
import os
import logging


def run_training(train_path: str = "artifacts/train.csv", test_path: str = "artifacts/test.csv"):
    # 1) data transformation
    dt = DataTransformation()
    train_arr, test_arr, preprocessor_path = dt.initiate_data_transformation(train_path, test_path)

    # 2) model training (returns dict with model)
    trainer = ModelTrainer()
    res = trainer.initiate_model_trainer(train_arr, test_arr)
    best_model = res.get("model")

    # 3) build full pipeline and save
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")

    preprocessor = load_object(preprocessor_path)
    full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_model)])

    save_object(file_path=os.path.join("artifacts", "model.pkl"), obj=full_pipeline)
    print("Saved full pipeline to artifacts/model.pkl")


if __name__ == "__main__":
    run_training()
