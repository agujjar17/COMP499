import os
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Define the input and output directories
input_normal_dir = './Lay_visualization/Normal'
input_abnormal_dir = './Lay_visualization/Abnormal'
output_normal_dir = 'output_normal'
output_abnormal_dir = 'output_abnormal'

# Ensure the output directories exist
os.makedirs(output_normal_dir, exist_ok=True)
os.makedirs(output_abnormal_dir, exist_ok=True)

# Define the wavelet and the number of decomposition levels
wavelet = 'haar'  # You can change the wavelet type
n_levels = 3

def compute_wavelet_statistics(image):
    # Perform the wavelet transformation
    coeffs = pywt.wavedec2(image, wavelet, level=n_levels)
    
    statistics = {}

    for level, c in enumerate(coeffs):
        flat_coeffs = np.hstack([sub_c.flatten() for sub_c in c])
        
        # Compute statistics for each level
        statistics[f'Level {level}'] = {
            'min': np.min(flat_coeffs),
            'max': np.max(flat_coeffs),
            'mean': np.mean(flat_coeffs),
            'std': np.std(flat_coeffs)
        }
    
    return statistics

def create_pdf(input_dir, output_dir, pdf_filename):
    pdf_pages = PdfPages(pdf_filename)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_image_path = os.path.join(input_dir, filename)

            # Load the image
            image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # Compute wavelet statistics
            wavelet_stats = compute_wavelet_statistics(image)

            # Display the original image
            plt.figure(figsize=(5, 5))
            plt.imshow(image, cmap='gray')
            plt.title('Original Image')
            pdf_pages.savefig()
            plt.close()

            # Print wavelet statistics
            print(f"Statistics for {filename}:")
            for level, stats in wavelet_stats.items():
                print(f"Level {level}:")
                for stat, value in stats.items():
                    print(f"{stat}: {value}")

    pdf_pages.close()

# Create a PDF for "normal" images
pdf_filename_normal = 'normal_images.pdf'
create_pdf(input_normal_dir, output_normal_dir, pdf_filename_normal)

# Create a PDF for "abnormal" images
pdf_filename_abnormal = 'abnormal_images.pdf'
create_pdf(input_abnormal_dir, output_abnormal_dir, pdf_filename_abnormal)
