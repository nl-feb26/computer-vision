#first install the library by this command 'pip install opencv-python mediapipe'


import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variables for respiration rate calculation
start_time = time.time()
breath_count = 0

def detect_breathing(frame, pose_landmarks):
    chest_point = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    left_hip_point = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
    right_hip_point = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y

    abdomen_point = (left_hip_point + right_hip_point) / 2
    distance = abdomen_point - chest_point

    threshold = 0.1

    if distance > threshold:
        return True
    else:
        return False

def calculate_respiration_rate():
    global breath_count, start_time
    elapsed_time = time.time() - start_time
    if elapsed_time >= 60:
        respiration_rate = breath_count / (elapsed_time / 60)
        print(f"Respiration Rate: {respiration_rate:.2f} breaths per minute")
        breath_count = 0
        start_time = time.time()  # Reset start time
        return respiration_rate
    else:
        return None

def main():
    global breath_count

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                breathing_detected = detect_breathing(frame, results.pose_landmarks)

                if breathing_detected:
                    breath_count += 1

                respiration_rate = calculate_respiration_rate()
                if respiration_rate is not None:
                    cv2.putText(frame, f"Respiration Rate: {respiration_rate:.2f} breaths/min", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Breathing Monitor', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
