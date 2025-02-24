import numpy as np
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    degree = np.abs(radian * 180 / np.pi)
    if degree > 180:
        degree = 360 - degree
    return degree


def calculate_position(x, y):
    return (int(x * 640), int(y * 480))


def get_joint_position(landmarks, joint):
    joint_map = {
        "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE.value,
        "LEFT_EAR": mp_pose.PoseLandmark.LEFT_EAR.value,
        "LEFT_ELBOW": mp_pose.PoseLandmark.LEFT_ELBOW.value,
        "LEFT_EYE_INNER": mp_pose.PoseLandmark.LEFT_EYE_INNER.value,
        "LEFT_EYE_OUTER": mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
        "LEFT_EYE": mp_pose.PoseLandmark.LEFT_EYE.value,
        "LEFT_FOOT_INDEX": mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
        "LEFT_HEEL": mp_pose.PoseLandmark.LEFT_HEEL.value,
        "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP.value,
        "LEFT_INDEX": mp_pose.PoseLandmark.LEFT_INDEX.value,
        "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE.value,
        "LEFT_PINKY": mp_pose.PoseLandmark.LEFT_PINKY.value,
        "LEFT_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        "LEFT_THUMB": mp_pose.PoseLandmark.LEFT_THUMB.value,
        "LEFT_WRIST": mp_pose.PoseLandmark.LEFT_WRIST.value,
        "MOUTH_LEFT": mp_pose.PoseLandmark.MOUTH_LEFT.value,
        "MOUTH_RIGHT": mp_pose.PoseLandmark.MOUTH_RIGHT.value,
        "NOSE": mp_pose.PoseLandmark.NOSE.value,
        "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        "RIGHT_EAR": mp_pose.PoseLandmark.RIGHT_EAR.value,
        "RIGHT_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        "RIGHT_EYE_INNER": mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,
        "RIGHT_EYE_OUTER": mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
        "RIGHT_EYE": mp_pose.PoseLandmark.RIGHT_EYE.value,
        "RIGHT_FOOT_INDEX": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
        "RIGHT_HEEL": mp_pose.PoseLandmark.RIGHT_HEEL.value,
        "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP.value,
        "RIGHT_INDEX": mp_pose.PoseLandmark.RIGHT_INDEX.value,
        "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE.value,
        "RIGHT_PINKY": mp_pose.PoseLandmark.RIGHT_PINKY.value,
        "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        "RIGHT_THUMB": mp_pose.PoseLandmark.RIGHT_THUMB.value,
        "RIGHT_WRIST": mp_pose.PoseLandmark.RIGHT_WRIST.value,
    }

    if joint in joint_map:
        joint_landmark = joint_map[joint]
        return (
            landmarks[joint_landmark].x,
            landmarks[joint_landmark].y,
            landmarks[joint_landmark].z,
        )
    else:
        raise ValueError(f"Invalid joint name: {joint}")


def draw_angle(image, angle, pos):

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)  # White color for the text
    thickness = 2
    lineType = cv2.LINE_AA

    cv2.putText(
        img=image,
        text=f"{angle} deg",
        org=pos,
        fontFace=fontFace,
        fontScale=fontScale,
        color=color,
        thickness=thickness,
        lineType=lineType,
    )


def draw_reps_counter(image, rep):

    text = f"{rep} reps"
    org = (10, 50)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)  # White color for the text
    thickness = 2
    lineType = cv2.LINE_AA
    (text_width, text_height), baseline = cv2.getTextSize(
        text, fontFace, fontScale, thickness
    )
    text_height += baseline
    cv2.rectangle(
        image,
        org,
        (org[0] + text_width, org[1] - text_height),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        img=image,
        text=text,
        org=org,
        fontFace=fontFace,
        fontScale=fontScale,
        color=color,
        thickness=thickness,
        lineType=lineType,
    )


def exercise_counter(exercise, angle, count, stage):
    
    if exercise == "curl":
        if angle >= 170:
            stage = "down"
            count = count
        elif angle <= 20 and stage == "down":
            stage = "up"
            count += 1
        return count, stage
