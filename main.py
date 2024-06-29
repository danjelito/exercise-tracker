import numpy as np
import cv2
import mediapipe as mp
from helper import (
    calculate_angle,
    calculate_position,
    get_joint_position,
    exercise_counter,
    draw_angle,
    draw_reps_counter,
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    frame_counter = 0
    exercise_name = "curl"
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
                left_a_point = get_joint_position(landmarks, "LEFT_SHOULDER")
                left_b_point = get_joint_position(landmarks, "LEFT_ELBOW")
                left_c_point = get_joint_position(landmarks, "LEFT_WRIST")
                right_a_point = get_joint_position(landmarks, "RIGHT_SHOULDER")
                right_b_point = get_joint_position(landmarks, "RIGHT_ELBOW")
                right_c_point = get_joint_position(landmarks, "RIGHT_WRIST")

                # calculate angle and position
                left_angle = round(
                    calculate_angle(left_a_point, left_b_point, left_c_point)
                )
                left_pos = calculate_position(left_b_point[0], left_b_point[1])
                right_angle = round(
                    calculate_angle(right_a_point, right_b_point, right_c_point)
                )
                right_pos = calculate_position(right_b_point[0], right_b_point[1])

        else:
            image = frame

        # EXERCISE COUNTER
        exercise_count, exercise_stage = exercise_counter(
            exercise=exercise_name,
            angle=((left_angle + right_angle) / 2),  # use both angle
            count=exercise_count,
            stage=exercise_stage,
        )

        # draw reps counter
        draw_reps_counter(image, exercise_count)

        # draw angle text in the image
        draw_angle(image, right_angle, right_pos)
        draw_angle(image, left_angle, left_pos)

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
