#for cair
import sys

# Add the path to the specific folder containing Python packages
specific_folder_path = r'/gpfs/home/agujjar/anaconda3/envs/myenv/lib'
sys.path.append(specific_folder_path)

# Now you can import modules and packages from the specific folder

####################
import os
import pydicom
import numpy as np
from scipy.stats import skew, kurtosis
import pywt
import pickle

# Define the input and output directories
input_root_dir = './images'
output_root_dir = './output_diacom_txt_files'

# Ensure the output directory exists
os.makedirs(output_root_dir, exist_ok=True)

# Define the wavelet and the number of decomposition levels
wavelet = 'haar'  # can change the wavelet type here
n_levels = 3

def compute_wavelet_statistics_and_coefficients(image):
    # Perform the wavelet transformation
    coeffs = pywt.wavedec2(image, wavelet, level=n_levels)

    statistics = {}

    for level, c in enumerate(coeffs):
        flat_coeffs = np.hstack([sub_c.flatten() for sub_c in c])

        # Compute additional statistics for each level
        statistics[f'Level {level}'] = {
            'min': np.min(flat_coeffs),
            'max': np.max(flat_coeffs),
            'mean': np.mean(flat_coeffs),
            'std': np.std(flat_coeffs),
            'skewness': skew(flat_coeffs),
            'kurtosis': kurtosis(flat_coeffs)
        }

    return statistics, coeffs

def save_statistics(input_dir, output_dir):
    for patient_folder in os.listdir(input_dir):
        patient_folder_path = os.path.join(input_dir, patient_folder)
        output_patient_folder = os.path.join(output_dir, patient_folder)

        if os.path.isdir(patient_folder_path):
            os.makedirs(output_patient_folder, exist_ok=True)

            for dicom_file in os.listdir(patient_folder_path):
                dicom_file_path = os.path.join(patient_folder_path, dicom_file)

                # Read DICOM file
                dcm = pydicom.dcmread(dicom_file_path)

                # Extract pixel data
                image = dcm.pixel_array

                # Compute wavelet statistics and coefficients
                wavelet_stats, coefficients = compute_wavelet_statistics_and_coefficients(image)

                # Create a text file for each DICOM file to store statistics and coefficients
                stats_file_path = os.path.join(
                    output_patient_folder, f'{os.path.splitext(dicom_file)[0]}.txt')
                with open(stats_file_path, 'w') as stats_file:
                    stats_file.write(f"Statistics for {dicom_file}:\n")
                    for level, stats in wavelet_stats.items():
                        stats_file.write(f"Level {level}:\n")
                        for stat, value in stats.items():
                            stats_file.write(f"{stat}: {value}\n")
                    stats_file.write("\nWavelet Coefficients:\n")
                    stats_file.write(str(coefficients))  # Write coefficients directly

# Save statistics and coefficients for DICOM images
save_statistics(input_root_dir, output_root_dir)
