import os
import sys
import traceback
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


# ──────────────────────────────────────────────────────────────────────────────
#  Prediction Pipeline
# ──────────────────────────────────────────────────────────────────────────────
class PredictPipeline:
    """
    Loads the fitted pre-processing object and trained model, applies the
    pre-processing to incoming data, and returns predictions.

    • By default, it expects the two pickle files in  ./artifacts/
    • You can override that location by exporting the environment variable
      ARTIFACTS_DIR before starting Flask, e.g.:

          # Windows (CMD / PowerShell)
          set ARTIFACTS_DIR=D:\First_Data_Science_Project\models
          python app.py

          # macOS / Linux
          export ARTIFACTS_DIR=/path/to/models
          python app.py
    """

    def __init__(self) -> None:
        # Folder that contains model.pkl and preprocessor.pkl
        self.artifacts_dir = os.environ.get("ARTIFACTS_DIR", "artifacts")

        # Full paths to files
        self.model_path = os.path.join(self.artifacts_dir, "model.pkl")
        self.preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────
    def _load_components(self):
        """
        Load the trained model and fitted pre-processor from disk.
        """
        print(f"Loading objects from: {self.artifacts_dir!r}")
        model = load_object(file_path=self.model_path)
        preprocessor = load_object(file_path=self.preprocessor_path)
        return model, preprocessor

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────
    def predict(self, features: pd.DataFrame):
        """
        Transform incoming sample(s) and output prediction(s).
        """
        try:
            model, preprocessor = self._load_components()

            print("Scaling features …")
            data_scaled = preprocessor.transform(features)

            print("Making prediction …")
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            # Print full traceback for easier debugging, then wrap
            traceback.print_exc()
            raise CustomException(e, sys) from e


# ──────────────────────────────────────────────────────────────────────────────
#  Data-collection helper
# ──────────────────────────────────────────────────────────────────────────────
class CustomData:
    """
    Gathers raw form inputs and converts them into a single-row DataFrame
    that matches the training schema expected by the pre-processor.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Return a one-row pandas DataFrame containing the user inputs.
        """
        try:
            data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            traceback.print_exc()
            raise CustomException(e, sys) from e
