import numpy as np
import cv2
import mediapipe as mp
import helper

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while True:
        # capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # detect pose
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:

            # extract landmarks and coordinates
            landmarks = results.pose_landmarks.landmark
            left_shoulder = helper.get_joint_position(landmarks, "LEFT_SHOULDER")
            left_elbow = helper.get_joint_position(landmarks, "LEFT_ELBOW")
            left_wrist = helper.get_joint_position(landmarks, "LEFT_WRIST")
            right_shoulder = helper.get_joint_position(landmarks, "RIGHT_SHOULDER")
            right_elbow = helper.get_joint_position(landmarks, "RIGHT_ELBOW")
            right_wrist = helper.get_joint_position(landmarks, "RIGHT_WRIST")

            # calculate angle and position
            left_elbow_angle = round(
                helper.calculate_angle(left_shoulder, left_elbow, left_wrist)
            )
            left_elbow_pos = helper.calculate_position(left_elbow[0], left_elbow[1])
            right_elbow_angle = round(
                helper.calculate_angle(right_shoulder, right_elbow, right_wrist)
            )
            right_elbow_pos = helper.calculate_position(right_elbow[0], right_elbow[1])

            # draw angle text in the image
            cv2.putText(
                img=image,
                text=f"{left_elbow_angle} deg",
                org=left_elbow_pos,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                img=image,
                text=f"{right_elbow_angle} deg",
                org=right_elbow_pos,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # render landmark
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

        # display the resulting frame
        cv2.imshow("press q to quit", image)

        # break if q is pressed
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
