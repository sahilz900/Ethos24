import cv2
import os
from mtcnn import MTCNN
from skimage.metrics import structural_similarity as ssim

def upscale_frame(frame):
    return frame  
def calculate_psnr(original_image, upscaled_image):
    """Calculate PSNR between two images."""
    return cv2.PSNR(original_image, upscaled_image)

def calculate_ssim(original_image, upscaled_image):
    """Calculate SSIM between two images."""
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    upscaled_gray = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    return ssim(original_gray, upscaled_gray)

def process_video(video_path, output_folder, original_folder, upscaled_folder):
    face_detector = MTCNN()

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(original_folder, exist_ok=True)
    os.makedirs(upscaled_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break 
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        original_output_path = os.path.join(original_folder, f"original_frame_{frame_count:04d}.jpg")
        cv2.imwrite(original_output_path, frame)

        results = face_detector.detect_faces(frame)

        if results is not None and len(results) > 0:
            for box in results:
                x1, y1, width, height = box['box']
                x2, y2 = x1 + width, y1 + height  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        upscaled_frame = upscale_frame(frame)
        upscaled_output_path = os.path.join(upscaled_folder, f"upscaled_frame_{frame_count:04d}.jpg")
        cv2.imwrite(upscaled_output_path, upscaled_frame)

        original_image = cv2.imread(original_output_path)
        upscaled_image = cv2.imread(upscaled_output_path)

        if original_image is not None and upscaled_image is not None:
            psnr_value = calculate_psnr(original_image, upscaled_image)
            ssim_value = calculate_ssim(original_image, upscaled_image)

            print(f'Frame {frame_count}: PSNR: {psnr_value} dB, SSIM: {ssim_value}')
        else:
            print(f"Error: Could not read images for frame {frame_count}.")

        cv2.imshow('Video', frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 
    print(f"Processed {frame_count} frames and saved to {output_folder}, {original_folder}, and {upscaled_folder}.")

video_path = r'C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_Round2\data\vid1.mp4'
output_folder = r'C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_Round2\cropped_faces'
original_folder = r'C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_Round2\original_frames'
upscaled_folder = r'C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\IIT_Round2\upscaled_frames'

process_video(video_path, output_folder, original_folder, upscaled_folder)
