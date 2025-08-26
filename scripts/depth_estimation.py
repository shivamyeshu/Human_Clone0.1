import os
import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image   # 👈 ADD THIS

# Load MiDaS model
model_type = "DPT_Large"  # other options: DPT_Hybrid, MiDaS_small
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# Prepare transforms
transform = Compose([
    Resize(384),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

def estimate_depth(image_path, output_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 🔥 Convert numpy → PIL before transform
    img_pil = Image.fromarray(img_rgb)

    input_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_img = (depth_norm * 255).astype(np.uint8)

    cv2.imwrite(output_path, depth_img)
    print(f"Saved depth map to {output_path}")

def main():
    input_dir = "../extracted_frames"
    output_dir = "../depth_maps"
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
    for frame_file in frame_files:
        input_path = os.path.join(input_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file.replace(".jpg", "_depth.png"))
        estimate_depth(input_path, output_path)

if __name__ == "__main__":
    main()
