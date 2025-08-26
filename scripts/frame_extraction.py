import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from video at `frame_rate` frames per second
    and save them in `output_dir`.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)

    count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"Extracted {saved_count} frames at {frame_rate} fps from {video_path}")

if __name__ == "__main__":
    video_file = "../input_videos/Repliworld.mp4"  # Change as per your uploaded video name
    output_folder = "../extracted_frames"
    extract_frames(video_file, output_folder, frame_rate=1)  # Extract 1 frame per second
