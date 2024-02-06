import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

csv = './INbreast.csv'
df = pd.read_csv(csv, sep=';')

def getCancer(patient_ID):
    patient_ID = int(patient_ID)
    filtered_df = df[df['File Name'] == patient_ID]
    print(filtered_df)
    if not filtered_df.empty:
        cancer = filtered_df['Bi-Rads'].iloc[0]
        print("cancer is",cancer)
        if (cancer == '1' or cancer == '2' or cancer == '3'):
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

# Load data from the "output_dicom_txt_files" directory for training
train_data_dir = './output_dicom_txt_files_db'
for txt_file in os.listdir(train_data_dir):
    patient_ID = txt_file.split('_')[0]
    cancer = getCancer(patient_ID)
    txt_file_path = os.path.join(train_data_dir, txt_file)
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

# Load data from the "test_output_diacom_txt_files" directory for testing
test_data_dir = './test_output_diacom_txt_files_db'
for txt_file in os.listdir(test_data_dir):
    patient_ID = txt_file.split('_')[0]
    cancer = getCancer(patient_ID)
    txt_file_path = os.path.join(test_data_dir, txt_file)
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
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()
    print(f"Confusion Matrix for {model.__class__.__name__}:\n", confusion_matrix(y_test, test_predictions))
    
    # Metrics
    accuracy = accuracy_score(y_test, test_predictions)
    
    # Handling division by zero for precision
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    
    f1score = 2 * (tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) != 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    recall = sensitivity
    
    print(f"{model.__class__.__name__} Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1score}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("\n")
