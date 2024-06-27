import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


def calculate_position(x, y):
    return (int(x * 640), int(y * 480))


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
            left_shoulder = (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
            )
            left_elbow = (
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
            )
            left_wrist = (
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
            )

            # calculate angle and position
            left_elbow_angle = round(
                calculate_angle(left_shoulder, left_elbow, left_wrist)
            )
            left_elbow_pos = calculate_position(left_elbow[0], left_elbow[1])

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
