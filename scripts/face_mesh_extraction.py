import cv2
import mediapipe as mp
import os
import json

mp_face_mesh = mp.solutions.face_mesh

def process_frame(image_path, output_debug_dir=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return None
    
    # Use relaxed thresholds + tracking
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})

        # 🔍 Debug visualization (optional)
        if output_debug_dir is not None:
            os.makedirs(output_debug_dir, exist_ok=True)
            h, w, _ = image.shape
            for lm in face_landmarks.landmark[0:50]:  # draw first 50 points
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
            debug_path = os.path.join(output_debug_dir, os.path.basename(image_path))
            cv2.imwrite(debug_path, image)

        return landmarks


def main():
    input_dir = "../extracted_frames"
    output_dir = "../face_mesh"
    debug_dir = "../debug_frames"  
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
    for frame_file in frame_files:
        frame_path = os.path.join(input_dir, frame_file)
        landmarks = process_frame(frame_path, output_debug_dir=debug_dir)
        if landmarks is not None:
            json_path = os.path.join(output_dir, frame_file.replace(".jpg", ".json"))
            with open(json_path, "w") as f:
                json.dump(landmarks, f)
            print(f" Processed and saved landmarks for {frame_file}")
        else:
            print(f" No face landmarks found in {frame_file}")


if __name__ == "__main__":
    main()
