import cv2
import openvino as ov
from pathlib import Path
import numpy as np
import os
from detector import Detector
from landmark_detector import LandmarkDetector
from helper import Helper
import collections
import time
import recognition_detector as Recognition

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
DRIVER_ACTION_RECOGNITION_MODEL_NAME = DIRECTORY_NAME + \
    "/driver-action-recognition-adas-0002"


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

# omz converter를 이용한 bin, xml file 만들기
if not Path(CONVERTED_OPEN_CLOSED_EYE_MODEL_NAME).exists():
    os.system(
        f'omz_converter --name {model_name} --precisions FP32 --download_dir {Path(DIRECTORY_NAME)} --output_dir {Path(DIRECTORY_NAME)}')


# 변환한 모델 사용
open_closed_eye_model = core.read_model(CONVERTED_OPEN_CLOSED_EYE_MODEL_NAME)
open_closed_eye_model_compiled = core.compile_model(
    open_closed_eye_model, 'AUTO')
open_closed_eye_detector = Detector(open_closed_eye_model_compiled)

vocab_file_path = Path(DRIVER_ACTION_RECOGNITION_MODEL_NAME +
                       "/driver_actions.txt")

with vocab_file_path.open(mode='r') as f:
    labels = [line.strip() for line in f]

print(labels[0:9], np.shape(labels))

recognition_model_encoder = core.read_model(DRIVER_ACTION_RECOGNITION_MODEL_NAME +
                                            '/driver-action-recognition-adas-0002-encoder/FP16/driver-action-recognition-adas-0002-encoder.xml')
recognition_model_encoder_compiled = core.compile_model(
    recognition_model_encoder, "AUTO")

recognition_model_decoder = core.read_model(DRIVER_ACTION_RECOGNITION_MODEL_NAME +
                                            '/driver-action-recognition-adas-0002-decoder/FP16/driver-action-recognition-adas-0002-decoder.xml')
recognition_model_decoder_compiled = core.compile_model(
    recognition_model_decoder, "AUTO")

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
helper = Helper()

height_en, width_en = list(recognition_model_encoder.input(0).shape)[2:]
frames2decode = list(recognition_model_decoder.input(0).shape)[0:][1]

size = height_en
sample_duration = frames2decode
processing_time = 0
processing_times = collections.deque()
encoder_output = []
decoded_labels = [0, 0, 0]
decoded_top_probs = [0, 0, 0]
cnt = 0
text_inference_template = "Infer Time:{Time:.1f}ms, {fps:.1f}FPS"
text_template = "{label},{conf:.2f}%"

while cap.isOpened() and cap2.isOpened():
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    if not ret:
        break

    scale = 1280 / max(frame.shape)

    equ = helper.get_histogram(frame)

    equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
    recog = frame2.copy()
    cv2.imshow("frame2", recog)
    if scale < 1:
        recog = cv2.resize(recog, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)

    if cnt % 2 == 0:
        (preprocessed, _) = Recognition.preprocessing(recog, size)

        start_time = time.time()

        encoder_output.append(Recognition.encoder(
            preprocessed, recognition_model_encoder_compiled))

        if len(encoder_output) == sample_duration:
            decoded_labels, decoded_top_probs = Recognition.decoder(
                encoder_output, recognition_model_decoder_compiled, labels)
            encoder_output = []

        stop_time = time.time()

        processing_times.append(stop_time - start_time)

        if len(processing_times) > 200:
            processing_times.popleft()

        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time

    for i in range(0, 3):
        display_text = text_template.format(
            label=decoded_labels[i],
            conf=decoded_top_probs[i] * 100,
        )
        Recognition.display_text_fnc(frame, display_text, i)
        if (decoded_labels[i] == 'Texting right' or decoded_labels[i] == 'Texting left') and decoded_top_probs[i] > 0.8:
            Recognition.display_text_fnc(
                frame, 'DO NOT USE CELL PHONE', 6, 100, (0, 0, 255))

    display_text = text_inference_template.format(
        Time=processing_time, fps=fps)
    Recognition.display_text_fnc(frame, display_text, 3)

    # press esc to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

    face_detect_result = face_detector.detect(equ)

    # filter out face detection results with confidence(detection[2]) < 0.5
    valid_detections = [detection for detection in face_detect_result[0]
                        [0] if detection[2] > FACE_DETECTION_THRESHOLD]
    # frame shape: height, width, channels. get height and width
    frame_h, frame_w = equ.shape[:2]

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
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)

            # crop face
            face = equ[y_min:y_max, x_min:x_max]

            # detect facial landmarks
            landmark_detect_result = facial_landmarks_detector.detect(face)
            left_eye, right_eye = facial_landmarks_detector.extract_eyes_from_output(
                face, landmark_detect_result)

            left_eye_detect_result = open_closed_eye_detector.detect(left_eye)
            right_eye_detect_result = open_closed_eye_detector.detect(
                right_eye)

            left_eye_open_prob = left_eye_detect_result[0][1][0][0]
            right_eye_open_prob = right_eye_detect_result[0][1][0][0]

            if left_eye_open_prob < OPEN_CLOSED_THRESHOLD and right_eye_open_prob < OPEN_CLOSED_THRESHOLD:
                cv2.putText(frame, 'Eyes Closed', (x_min, y_max + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 0), 2)
                EYES_CLOSED_COUNTER += 1
            else:
                cv2.putText(frame, 'Eyes Open', (x_min, y_max + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

            # State에 맞게 Text값 조정
            cv2.putText(frame, focus_state[state], (x_min, y_min),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
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

        # print(EYES_CLOSED_COUNTER)
        cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
