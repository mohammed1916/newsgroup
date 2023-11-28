import os
import joblib

# Set the path to the models folder
models_folder = "/A:/Dev/FDS/newsgroup/models"

# Load the trained model
model_path = os.path.join(models_folder, "trained_model.joblib")
model = joblib.load(model_path)

# Load the dataset for prediction
dataset_path = "/A:/Dev/FDS/newsgroup/dataset.csv"
dataset = load_dataset(dataset_path)

# Perform prediction
predictions = model.predict(dataset)

# Print the predictions
print(predictions)
