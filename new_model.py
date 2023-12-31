import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

csv = './finding_annotations.csv'
df = pd.read_csv(csv)


def getCancer(patient_ID):
    filtered_df = df[df['image_id'] == patient_ID]
    if not filtered_df.empty:
        cancer = filtered_df['breast_birads'].iloc[0]
        if (cancer == 'BI-RADS 5' or cancer == 'BI-RADS 4'):
            index = 1
        else:
            index = 0
        return index


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

# Load data from the "train_output_diacom_files" directory for training
train_data_dir = './output_diacom_txt_files'
for folder in os.listdir(train_data_dir):
    folder_path = os.path.join(train_data_dir, folder)
    if os.path.isdir(folder_path):
        for txt_file in os.listdir(folder_path):
            patient_ID = txt_file.split('.')[0]
            cancer = getCancer(patient_ID)
            txt_file_path = os.path.join(folder_path, txt_file)
            statistics = extract_statistics_from_text(txt_file_path)

            # Add statistics to the feature vector
            feature_vector = [statistics['min'], statistics['max'],
                              statistics['mean'], statistics['std'],
                              statistics['skewness'], statistics['kurtosis']]
            data.append(feature_vector)

            # Assign labels (0 for 'Normal' and 1 for 'Abnormal')
            if cancer == 1:
                labels.append(1)
            else:
                labels.append(0)

# Load data from the "test_output_diacom_files" directory for testing
test_data_dir = './test_output_diacom_txt_files'
for folder in os.listdir(test_data_dir):
    folder_path = os.path.join(test_data_dir, folder)
    if os.path.isdir(folder_path):
        for txt_file in os.listdir(folder_path):
            patient_ID = txt_file.split('.')[0]
            cancer = getCancer(patient_ID)
            txt_file_path = os.path.join(folder_path, txt_file)
            statistics = extract_statistics_from_text(txt_file_path)

            # Add statistics to the feature vector
            feature_vector = [statistics['min'], statistics['max'],
                              statistics['mean'], statistics['std'],
                              statistics['skewness'], statistics['kurtosis']]
            data.append(feature_vector)

            # Assign labels (0 for 'Normal' and 1 for 'Abnormal')
            if cancer == 1:
                labels.append(1)
            else:
                labels.append(0)

print("Number of samples: ", len(data))
print("Number of 'Abnormal' samples (cancer):", labels.count(1))
print("Number of 'Normal' samples (no cancer):", labels.count(0))

# No need to split the data manually since you have separate folders for training and testing

# Define a list of machine learning algorithms to try
algorithms = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
    DecisionTreeClassifier()
]

# Split the data into training and testing sets
split_index = int(0.8 * len(data))
X_train, y_train = data[:split_index], labels[:split_index]
X_test, y_test = data[split_index:], labels[split_index:]

for model in algorithms:
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)
    model_name = model.__class__.__name__
    print(f"{model_name} Accuracy on Test Set: {accuracy}")
    print(f"Classification Report for {model_name}:\n", classification_report(y_test, test_predictions))
    print(f"Confusion Matrix for {model_name}:\n", confusion_matrix(y_test, test_predictions))
