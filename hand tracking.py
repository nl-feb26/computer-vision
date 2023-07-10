#step 1: importing libraries
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#step 2: creating and image input and processing it
while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
#step 3: working with each hand
    # checking whether a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

#Step 4 - Drawing the hand landmarks and hand connections on the hand image
                if id == 20:
                    cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

#step 5: displaying the output
    cv2.imshow("Output", image)
    cv2.waitKey(1)
