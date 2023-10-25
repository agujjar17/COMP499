# # from each wavelet map in the decomposition, four statistical feature were measured: mean intensity, standard deviation of intensity, skewness of intensity and kurtosis of intensity

import os
import cv2
import pywt
import numpy as np
from scipy.stats import skew, kurtosis

# Define the input directories
input_normal_dir = './Lay_visualization/Normal'
input_abnormal_dir = './Lay_visualization/Abnormal'
output_normal_dir = 'output_normal'
output_abnormal_dir = 'output_abnormal'

# Ensure the output directories exist
os.makedirs(output_normal_dir, exist_ok=True)
os.makedirs(output_abnormal_dir, exist_ok=True)

# Define the wavelet and the number of decomposition levels
wavelet = 'haar'  # can change the wavelet type here
n_levels = 3


def compute_wavelet_statistics(image):
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

    return statistics


def save_statistics(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_image_path = os.path.join(input_dir, filename)

            # Load the image
            image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # Compute wavelet statistics
            wavelet_stats = compute_wavelet_statistics(image)

            # Create a text file for each image to store statistics
            stats_file_path = os.path.join(
                output_dir, f'{os.path.splitext(filename)[0]}.txt')
            with open(stats_file_path, 'w') as stats_file:
                stats_file.write(f"Statistics for {filename}:\n")
                for level, stats in wavelet_stats.items():
                    stats_file.write(f"Level {level}:\n")
                    for stat, value in stats.items():
                        stats_file.write(f"{stat}: {value}\n")


# Save statistics for "normal" images
save_statistics(input_normal_dir, output_normal_dir)

# Save statistics for "abnormal" images
save_statistics(input_abnormal_dir, output_abnormal_dir)
