import cv2
from matplotlib import pyplot as plt
import numpy as np


class Helper:
    @staticmethod
    def get_histogram(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(gray)

        return cl1
