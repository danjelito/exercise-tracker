import numpy as np
import cv2 
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print("Cannot open camera")
	exit()

# setup mediapipe
with mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
) as pose:

    while True:
        # capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # detect pose
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # render detection
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # display the resulting frame
        cv2.imshow("press q to quit", image)

        # break if q is pressed
        if cv2.waitKey(1) == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()