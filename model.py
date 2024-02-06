#for cair
import sys

# Add the path to the specific folder containing Python packages
specific_folder_path = r'/gpfs/home/agujjar/anaconda3/envs/myenv/lib'
sys.path.append(specific_folder_path)

# Now you can import modules and packages from the specific folder

####################

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Function to extract statistics from a text file
def extract_statistics_from_text(text_file):
    statistics = {'min': 0.0, 'max': 0.0, 'mean': 0.0,
                  'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}

    with open(text_file, 'r') as stats_file:
        lines = stats_file.readlines()
        for line in lines:
            for key in statistics:
                if key in line:
                    value = float(line.split(":")[1])
                    statistics[key] = value

    return statistics

# Load and preprocess the data
data = []
labels = []

# Load data from the "output_diacom_txt_files" directory
data_dir = 'output_diacom_images'
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        for txt_file in os.listdir(folder_path):
            txt_file_path = os.path.join(folder_path, txt_file)
            statistics = extract_statistics_from_text(txt_file_path)

            # Add statistics to the feature vector
            feature_vector = [statistics['min'], statistics['max'],
                              statistics['mean'], statistics['std'],
                              statistics['skewness'], statistics['kurtosis']]
            data.append(feature_vector)

            # Assign labels (0 for 'Normal' and 1 for 'Abnormal')
            if folder == "Abnormal":
                labels.append(1)
            else:
                labels.append(0)

print("Number of samples: ", len(data))
print("Number of 'Abnormal' samples (cancer):", labels.count(1))
print("Number of 'Normal' samples (no cancer):", labels.count(0))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# Define a list of machine learning algorithms to try
algorithms = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
    DecisionTreeClassifier()
]

for model in algorithms:
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)
    model_name = model.__class__.__name__
    print(f"{model_name} Accuracy on Test Set: {accuracy}")
    print(f"Classification Report for {model_name}:\n", classification_report(y_test, test_predictions))
    print(f"Confusion Matrix for {model_name}:\n", confusion_matrix(y_test, test_predictions))

# Make predictions for a new folder containing 4 DICOM files
# new_patient_data = []
# for dicom_file in os.listdir("output_diacom_images/ff9e4f3aa02c617fd62e2ec6782b6246"):
#     dicom_file_path = os.path.join(
#         "output_diacom_images/ff9e4f3aa02c617fd62e2ec6782b6246", dicom_file)
#     statistics = extract_statistics_from_text(dicom_file_path)
#     feature_vector = [statistics['min'], statistics['max'],
#                       statistics['mean'], statistics['std'],
#                       statistics['skewness'], statistics['kurtosis']]
#     new_patient_data.append(feature_vector)

# for model in algorithms:
#     new_predictions = model.predict(new_patient_data)
#     model_name = model.__class__.__name__
#     if any(new_predictions == 1):
#         print(f"{model_name}: The patient has cancer.")
#     else:
#         print(f"{model_name}: The patient does not have cancer.")

