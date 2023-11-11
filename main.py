#first install the library by this command 'pip install opencv-python mediapipe'


import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def detect_breathing(frame, pose_landmarks):
    # Extract key points for the chest and abdomen
    chest_point = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    left_hip_point = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
    right_hip_point = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y

    # Calculate the average hip point
    abdomen_point = (left_hip_point + right_hip_point) / 2

    # Calculate the distance between chest and abdomen points
    distance = abdomen_point - chest_point

    # You may need to fine-tune this threshold based on your use case
    threshold = 0.1

    # Check if the distance exceeds the threshold
    if distance > threshold:
        return True
    else:
        return False

def main():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(rgb_frame)

            # If landmarks are detected, draw them on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Detect breathing
                breathing_detected = detect_breathing(frame, results.pose_landmarks)

                # Display the result
                cv2.putText(frame, f"Breathing Detected: {breathing_detected}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Breathing Monitor', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the VideoCapture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

