from detector import Detector
import numpy as np

class LandmarkDetector(Detector):
    def __init__(self, compiled_model):
        super().__init__(compiled_model)

    def extract_eye_coordinates(self, result):
        output = result[0]
        # output에서 x와 y 좌표를 분리하여 35개의 (x, y) 쌍을 만듭니다.
        keypoints = [(output[i], output[i + 1])
                     for i in range(0, len(output), 2)]
        return keypoints

    def crop_eye_region(self, image, p1, p2, expansion=1.8):
        # 이미지 크기를 기준으로 키포인트의 좌표를 확장합니다.
        p1 = (int(p1[0] * image.shape[1]), int(p1[1] * image.shape[0]))
        p2 = (int(p2[0] * image.shape[1]), int(p2[1] * image.shape[0]))

        # 두 키포인트 사이의 중심을 찾습니다.
        center_x = (p1[0] + p2[0]) / 2
        center_y = (p1[1] + p2[1]) / 2

        # 두 키포인트 사이의 거리를 계산합니다.
        dist = np.linalg.norm(np.array(p1) - np.array(p2))

        # 거리를 계산한 후, expansion을 곱하여 확장된 사각형의 half_width를 계산합니다.
        half_width = dist * expansion / 2

        # 두 키포인트 사이의 거리를 기준으로 확장된 사각형의 좌상단, 우하단 좌표를 계산합니다.
        top_left = (int(center_x - half_width), int(center_y - half_width))
        bottom_right = (int(center_x + half_width), int(center_y + half_width))

        return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    def extract_eyes_from_output(self, image, output):
        keypoints = self.extract_eye_coordinates(output)

        # [Left Eye] p0, p1: corners of the eye, located on the boundary of the eyeball and the eyelid.
        left_eye = self.crop_eye_region(image, keypoints[0], keypoints[1])

        # [Right Eye] p2, p3: corners of the eye, located on the boundary of the eyeball and the eyelid.
        right_eye = self.crop_eye_region(image, keypoints[2], keypoints[3])

        return left_eye, right_eye
