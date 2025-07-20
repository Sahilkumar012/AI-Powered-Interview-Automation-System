# final model for face/head and eye tracking
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")  # Use a face-trained model if you have one

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        exit()

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    def get_head_pose(image, landmarks):
        image_points = np.array([
            landmarks[1],    # Nose tip
            landmarks[152],  # Chin
            landmarks[33],   # Right eye outer corner
            landmarks[263],  # Left eye outer corner
            landmarks[78],   # Right mouth corner
            landmarks[308]   # Left mouth corner
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (225.0, 170.0, -135.0),      # Right eye outer corner
            (-225.0, 170.0, -135.0),     # Left eye outer corner
            (150.0, -150.0, -125.0),     # Right mouth corner
            (-150.0, -150.0, -125.0)     # Left mouth corner
        ])

        focal_length = image.shape[1]
        center = (image.shape[1] / 2, image.shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))
        success, rvec, _, _ = cv2.solvePnPRansac(
            model_points, image_points, camera_matrix, dist_coeffs)

        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, np.zeros((3, 1))))
        _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(proj_matrix)

        return angles  # pitch, yaw, roll

    # ✅ Updated eye direction using iris tracking
    def get_eye_direction(landmarks, iw):
        try:
            right_iris_center = landmarks[468]  # Iris center
            right_eye_inner = landmarks[133]
            right_eye_outer = landmarks[33]

            eye_width = right_eye_outer[0] - right_eye_inner[0]
            if eye_width == 0:
                return "Undetected"

            iris_relative_pos = (right_iris_center[0] - right_eye_inner[0]) / eye_width

            if iris_relative_pos < 0.40:
                return "Looking Left"
            elif iris_relative_pos > 0.60:
                return "Looking Right"
            else:
                return "Looking Center"
        except:
            return "Undetected"

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)

        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            result_mesh = face_mesh.process(face_rgb)

            if result_mesh.multi_face_landmarks:
                for landmarks in result_mesh.multi_face_landmarks:
                    ih, iw, _ = face.shape
                    coords = [(p.x * iw, p.y * ih) for p in landmarks.landmark]

                    try:
                        angles = get_head_pose(face, coords)
                        pitch, yaw, roll = [a[0] for a in angles]

                        if abs(yaw) > 10:  # ✅ Increased yaw threshold for more reliable detection
                            cheat = "⚠️ Head turned!"
                            print(f"Yaw: {yaw:.2f} - Cheat: {cheat}")
                        elif abs(pitch) > 7:
                            cheat = "⚠️ Looking up/down!"
                            print(f"Pitch: {pitch:.2f} - Cheat: {cheat}")
                        else:
                            cheat = ""
                    except:
                        cheat = "Head Pose Error"
                        print("Head Pose Error")

                    eye_dir = get_eye_direction(coords, iw)
                    if eye_dir != "Looking Center":
                        cheat = f"⚠️ {eye_dir}"
                        print(f"Eye Direction: {eye_dir} - Cheat: {cheat}")
                    else:
                        cheat = ""
                        print("Eye Direction: Looking Center")

                    # Draw box and alert
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, eye_dir, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    if cheat:
                        cv2.putText(frame, cheat, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("YOLO + MediaPipe Cheat Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Add this at the very bottom:
if __name__ == "__main__":
    main()
