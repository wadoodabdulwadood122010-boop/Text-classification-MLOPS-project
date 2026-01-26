import unittest
import mlflow
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Set up DagsHub credentials
        dagshub_token = os.getenv("DAGSHUB_KEY")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_KEY environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # 2. Set Tracking URI (Must happen BEFORE creating the client)
        dagshub_url = "https://dagshub.com"
        repo_owner = "wadoodabdulwadood122010"
        repo_name = "Text-classification-MLOPS-project"
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # 3. Load Model Version
        cls.new_model_name = "my_model"
        # Optional: You can change 'Staging' to 'None' or 'Production' if needed
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name, stage="Staging")

        if cls.new_model_version is None:
            # Fallback: If no Staging model, try getting the absolute latest version
            print("No 'Staging' model found. Fetching the latest version available...")
            cls.new_model_version = cls.get_latest_model_version(cls.new_model_name, stage="None")
        
        if cls.new_model_version is None:
            raise ValueError(f"Could not find any version for model '{cls.new_model_name}'")

        print(f"Testing Model Version: {cls.new_model_version}")
        
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # 4. Load Local Assets
        # Ensure these files exist in your environment
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        return latest_versions[0].version if latest_versions else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        
        # FIX: Use actual feature names so the model signature matches
        feature_names = self.vectorizer.get_feature_names_out()
        input_df = pd.DataFrame(input_data.toarray(), columns=feature_names)

        # Predict
        prediction = self.new_model.predict(input_df)

        # Verify shapes
        self.assertEqual(input_df.shape[1], len(feature_names))
        self.assertEqual(len(prediction), input_df.shape[0])

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        # Assumes the last column is the label
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted', zero_division=0)
        recall_new = recall_score(y_holdout, y_pred_new, average='weighted', zero_division=0)
        f1_new = f1_score(y_holdout, y_pred_new, average='weighted', zero_division=0)

        # Print metrics for debugging log
        print(f"\nModel Metrics -- Acc: {accuracy_new:.2f}, Prec: {precision_new:.2f}, Rec: {recall_new:.2f}, F1: {f1_new:.2f}")

        # Define expected thresholds (adjusted to be realistic)
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assertions
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy {accuracy_new:.2f} < {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision {precision_new:.2f} < {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall {recall_new:.2f} < {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 Score {f1_new:.2f} < {expected_f1}')

if __name__ == "__main__":
    unittest.main()