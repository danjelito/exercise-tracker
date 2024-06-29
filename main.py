import numpy as np
import cv2
import mediapipe as mp
from helper import calculate_angle, calculate_position, get_joint_position, curl_counter

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    frame_counter = 0
    exercise_count = 0
    exercise_stage = None

    while True:
        # capture frame-by-frame
        ret, frame = cap.read()
        frame_counter += 1

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # detect pose
        # do it every 2 frames
        if frame_counter % 2 == 0 or frame_counter == 1:

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:

                # extract landmarks and coordinates
                landmarks = results.pose_landmarks.landmark
                left_shoulder = get_joint_position(landmarks, "LEFT_SHOULDER")
                left_elbow = get_joint_position(landmarks, "LEFT_ELBOW")
                left_wrist = get_joint_position(landmarks, "LEFT_WRIST")
                right_shoulder = get_joint_position(landmarks, "RIGHT_SHOULDER")
                right_elbow = get_joint_position(landmarks, "RIGHT_ELBOW")
                right_wrist = get_joint_position(landmarks, "RIGHT_WRIST")

                # calculate angle and position
                left_elbow_angle = round(
                    calculate_angle(left_shoulder, left_elbow, left_wrist)
                )
                left_elbow_pos = calculate_position(left_elbow[0], left_elbow[1])
                right_elbow_angle = round(
                    calculate_angle(right_shoulder, right_elbow, right_wrist)
                )
                right_elbow_pos = calculate_position(right_elbow[0], right_elbow[1])
        else:
            image = frame

        # EXERCISE COUNTER
        if frame_counter >= 2:
            exercise_count, exercise_stage = curl_counter(
                right_elbow_angle, exercise_count, exercise_stage
            )
            cv2.putText(
                img=image,
                text=f"{exercise_count} reps",
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # draw angle text in the image
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
