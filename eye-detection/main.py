import cv2
import openvino as ov
from pathlib import Path
import numpy as np
import os
from detector import Detector
from landmark_detector import LandmarkDetector

OPEN_CLOSED_THRESHOLD = 0.7
FACE_DETECTION_THRESHOLD = 0.5
EYES_CLOSED_COUNTER_THRESHOLD = 5
EYES_CLOSED_COUNTER = 0
COUNTER = 0

DIRECTORY_NAME = "model"
FACIAL_DETECTION_MODEL_NAME = DIRECTORY_NAME + \
    "/face-detection-adas-0001/FP32/face-detection-adas-0001.xml"
FACIAL_LANDMARKS_MODEL_NAME = DIRECTORY_NAME + \
    "/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml"
CONVERTED_OPEN_CLOSED_EYE_MODEL_NAME = DIRECTORY_NAME + \
    "/public/open-closed-eye-0001/FP32/open-closed-eye-0001.xml"

focus_state = ['Good', 'Warning', 'Bad']
state = 0
core = ov.Core()
face_detection_model = core.read_model(FACIAL_DETECTION_MODEL_NAME)
face_detection_model_compiled = core.compile_model(
    face_detection_model, 'AUTO')
face_detector = Detector(face_detection_model_compiled)


facial_landmarks_model = core.read_model(FACIAL_LANDMARKS_MODEL_NAME)
facial_landmarks_model_compiled = core.compile_model(
    facial_landmarks_model, 'AUTO')
facial_landmarks_detector = LandmarkDetector(facial_landmarks_model_compiled)

model_name = 'open-closed-eye-0001'

#omz converter를 이용한 bin, xml file 만들기
if not Path(CONVERTED_OPEN_CLOSED_EYE_MODEL_NAME).exists():
    os.system(f'omz_converter --name {model_name} --precisions FP32 --download_dir {Path(DIRECTORY_NAME)} --output_dir {Path(DIRECTORY_NAME)}')


# 변환한 모델 사용
open_closed_eye_model = core.read_model(CONVERTED_OPEN_CLOSED_EYE_MODEL_NAME)
open_closed_eye_model_compiled = core.compile_model(
    open_closed_eye_model, 'AUTO')
open_closed_eye_detector = Detector(open_closed_eye_model_compiled)

cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    contrast = 0.8
    brightness = -20
    frame[:,:,2] = np.clip(contrast * frame[:,:,2] + brightness, 0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    # press esc to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

    cv2.putText(frame, 'Press ESC to quit', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    face_detect_result = face_detector.detect(frame)

    # filter out face detection results with confidence(detection[2]) < 0.5
    valid_detections = [detection for detection in face_detect_result[0]
                        [0] if detection[2] > FACE_DETECTION_THRESHOLD]
    # frame shape: height, width, channels. get height and width
    frame_h, frame_w = frame.shape[:2]

    if len(valid_detections) == 0:
        cv2.putText(frame, 'No Face Detected', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        continue

    for detection in valid_detections:
        image_id, label, conf, x_min, y_min, x_max, y_max = detection
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        x_min = int(x_min * frame_w)
        y_min = int(y_min * frame_h)
        x_max = int(x_max * frame_w)
        y_max = int(y_max * frame_h)

        if x_max - x_min > 100:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # crop face
            face = frame[y_min:y_max, x_min:x_max]

            # detect facial landmarks
            landmark_detect_result = facial_landmarks_detector.detect(face)
            left_eye, right_eye = facial_landmarks_detector.extract_eyes_from_output(
                face, landmark_detect_result)

            left_eye_detect_result = open_closed_eye_detector.detect(left_eye)
            right_eye_detect_result = open_closed_eye_detector.detect(right_eye)

            left_eye_open_prob = left_eye_detect_result[0][1][0][0]
            right_eye_open_prob = right_eye_detect_result[0][1][0][0]

            if left_eye_open_prob < OPEN_CLOSED_THRESHOLD and right_eye_open_prob < OPEN_CLOSED_THRESHOLD:
                cv2.putText(frame, 'Eyes Closed', (x_min, y_max + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 0), 2)
                EYES_CLOSED_COUNTER += 1
            else:
                cv2.putText(frame, 'Eyes Open', (x_min, y_max + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

            #State에 맞게 Text값 조정
            cv2.putText(frame, focus_state[state], (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            COUNTER += 1
            if COUNTER > 100:
                if EYES_CLOSED_COUNTER < 20:
                    state = 0
                elif EYES_CLOSED_COUNTER < 50:
                    state = 1
                else:
                    state = 2
                EYES_CLOSED_COUNTER = 0
                COUNTER = 0

        #print(EYES_CLOSED_COUNTER)
        cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
