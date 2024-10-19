import cv2
import numpy as np
import os

def calculate_psnr(original, compared):
    """Calculate PSNR between two images."""
    mse = np.mean((original - compared) ** 2)
    if mse == 0:
        return float('inf')  
    max_pixel = 255.0 
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compare_images(original_folder, processed_folder):
    """Compare images in two folders and return metrics."""
    original_files = sorted(os.listdir(original_folder))
    processed_files = sorted(os.listdir(processed_folder))

    if len(original_files) != len(processed_files):
        raise ValueError("Number of images in both folders must be the same.")

    psnr_values = []

    for original_file, processed_file in zip(original_files, processed_files):
        original_path = os.path.join(original_folder, original_file)
        processed_path = os.path.join(processed_folder, processed_file)

        original_image = cv2.imread(original_path)
        processed_image = cv2.imread(processed_path)

        processed_image_resized = cv2.resize(processed_image, (original_image.shape[1], original_image.shape[0]))

        psnr_value = calculate_psnr(original_image, processed_image_resized)
        psnr_values.append(psnr_value)

    return psnr_values

def main():
    original_folder = r'C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_Round2\original_frames'
    upscaled_folder = r'C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_Round2\upscaled_frames'
    cropped_faces_folder = r'C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_Round2\cropped_faces'

    upscaled_metrics = compare_images(original_folder, upscaled_folder)
    print("PSNR values for upscaled frames:")
    for i, psnr in enumerate(upscaled_metrics):
        print(f"Image {i + 1}: PSNR = {psnr:.2f} dB")

    cropped_metrics = compare_images(original_folder, cropped_faces_folder)
    print("PSNR values for cropped faces:")
    for i, psnr in enumerate(cropped_metrics):
        print(f"Image {i + 1}: PSNR = {psnr:.2f} dB")

if __name__ == "__main__":
    main()
