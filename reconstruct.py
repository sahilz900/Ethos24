import os
import cv2
import numpy as np
import torch
from mtcnn import MTCNN

class RealESRGANModel(torch.nn.Module):
    def __init__(self):
        super(RealESRGANModel, self).__init__()
        pass

    def forward(self, x):
        return x 
class RealESRGAN:
    def __init__(self, model_path):
        self.model = RealESRGANModel() 
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)  # Load weights
        self.model.eval() 
    def upscale(self, img):
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        with torch.no_grad():
            output_tensor = self.model(img_tensor)

        output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255
        return np.clip(output_image, 0, 255).astype(np.uint8)  
def calculate_psnr(original, upscaled):
    """Calculate PSNR between the original and upscaled images."""
    mse = np.mean((original - upscaled) ** 2)
    return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def process_video(video_path, cropped_faces_folder, original_frames_folder, upscaled_folder, model):
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        original_frame_path = os.path.join(original_frames_folder, f"original_frame_{frame_count}.jpg")
        cv2.imwrite(original_frame_path, frame)

        for face in faces:
            x, y, width, height = face['box']
            cropped_face = frame[y:y + height, x:x + width]

            face_filename = f"cropped_face_{frame_count}.jpg"
            cv2.imwrite(os.path.join(cropped_faces_folder, face_filename), cropped_face)

            upscaled_face = model.upscale(cropped_face)
            upscaled_face_filename = f"upscaled_face_{frame_count}.jpg"
            cv2.imwrite(os.path.join(upscaled_folder, upscaled_face_filename), upscaled_face)

            psnr_value = calculate_psnr(cropped_face, upscaled_face)
            print(f"PSNR for frame {frame_count}: {psnr_value:.2f} dB")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
