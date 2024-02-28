import cv2
import numpy as np
from openvino.runtime.ie_api import CompiledModel
from typing import Tuple, List


def center_crop(frame: np.ndarray) -> np.ndarray:
    """
    Center crop squared the original frame to standardize the input image to the encoder model

    :param frame: input frame
    :returns: center-crop-squared frame
    """
    img_h, img_w, _ = frame.shape
    min_dim = min(img_h, img_w)
    start_x = int((img_w - min_dim) / 2.0)
    start_y = int((img_h - min_dim) / 2.0)
    roi = [start_y, (start_y + min_dim), start_x, (start_x + min_dim)]
    return frame[start_y: (start_y + min_dim), start_x: (start_x + min_dim), ...], roi


def adaptive_resize(frame: np.ndarray, size: int) -> np.ndarray:
    """
    The frame going to be resized to have a height of size or a width of size

    :param frame: input frame
    :param size: input size to encoder model
    :returns: resized frame, np.array type
    """
    h, w, _ = frame.shape
    scale = size / min(h, w)
    w_scaled, h_scaled = int(w * scale), int(h * scale)
    if w_scaled == w and h_scaled == h:
        return frame
    return cv2.resize(frame, (w_scaled, h_scaled))


def decode_output(probs: np.ndarray, labels: np.ndarray, top_k: int = 3) -> np.ndarray:
    """
    Decodes top probabilities into corresponding label names

    :param probs: confidence vector for 400 actions
    :param labels: list of actions
    :param top_k: The k most probable positions in the list of labels
    :returns: decoded_labels: The k most probable actions from the labels list
            decoded_top_probs: confidence for the k most probable actions
    """
    top_ind = np.argsort(-1 * probs)[:top_k]
    out_label = np.array(labels)[top_ind.astype(int)]
    decoded_labels = [out_label[0][0], out_label[0][1], out_label[0][2]]
    top_probs = np.array(probs)[0][top_ind.astype(int)]
    decoded_top_probs = [top_probs[0][0], top_probs[0][1], top_probs[0][2]]
    return decoded_labels, decoded_top_probs


def rec_frame_display(frame: np.ndarray, roi) -> np.ndarray:
    """
    Draw a rec frame over actual frame

    :param frame: input frame
    :param roi: Region of interest, image section processed by the Encoder
    :returns: frame with drawed shape

    """

    cv2.line(frame, (roi[2] + 3, roi[0] + 3),
             (roi[2] + 3, roi[0] + 100), (0, 200, 0), 2)
    cv2.line(frame, (roi[2] + 3, roi[0] + 3),
             (roi[2] + 100, roi[0] + 3), (0, 200, 0), 2)
    cv2.line(frame, (roi[3] - 3, roi[1] - 3),
             (roi[3] - 3, roi[1] - 100), (0, 200, 0), 2)
    cv2.line(frame, (roi[3] - 3, roi[1] - 3),
             (roi[3] - 100, roi[1] - 3), (0, 200, 0), 2)
    cv2.line(frame, (roi[3] - 3, roi[0] + 3),
             (roi[3] - 3, roi[0] + 100), (0, 200, 0), 2)
    cv2.line(frame, (roi[3] - 3, roi[0] + 3),
             (roi[3] - 100, roi[0] + 3), (0, 200, 0), 2)
    cv2.line(frame, (roi[2] + 3, roi[1] - 3),
             (roi[2] + 3, roi[1] - 100), (0, 200, 0), 2)
    cv2.line(frame, (roi[2] + 3, roi[1] - 3),
             (roi[2] + 100, roi[1] - 3), (0, 200, 0), 2)
    # Write ROI over actual frame
    FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
    org = (roi[2] + 3, roi[1] - 3)
    org2 = (roi[2] + 2, roi[1] - 2)
    FONT_SIZE = 0.5
    FONT_COLOR = (0, 200, 0)
    FONT_COLOR2 = (0, 0, 0)
    cv2.putText(frame, "ROI", org2, FONT_STYLE, FONT_SIZE, FONT_COLOR2)
    cv2.putText(frame, "ROI", org, FONT_STYLE, FONT_SIZE, FONT_COLOR)
    return frame


def display_text_fnc(frame: np.ndarray, display_text: str, index: int, left: int = 0, color: tuple = (255, 255, 255)):
    """
    Include a text on the analyzed frame

    :param frame: input frame
    :param display_text: text to add on the frame
    :param index: index line dor adding text

    """
    # Configuration for displaying images with text.
    FONT_COLOR = color
    FONT_COLOR2 = (0, 0, 0)
    FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX
    FONT_SIZE = 0.7
    TEXT_VERTICAL_INTERVAL = 25
    TEXT_LEFT_MARGIN = 15 + left
    # ROI over actual frame
    (processed, roi) = center_crop(frame)
    # Draw a ROI over actual frame.
    frame = rec_frame_display(frame, roi)
    # Put a text over actual frame.
    text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL * (index + 1))
    text_loc2 = (TEXT_LEFT_MARGIN + 1,
                 TEXT_VERTICAL_INTERVAL * (index + 1) + 1)
    cv2.putText(frame, display_text, text_loc2,
                FONT_STYLE, FONT_SIZE, FONT_COLOR2)
    cv2.putText(frame, display_text, text_loc,
                FONT_STYLE, FONT_SIZE, FONT_COLOR)


def preprocessing(frame: np.ndarray, size: int) -> np.ndarray:
    """
    Preparing frame before Encoder.
    The image should be scaled to its shortest dimension at "size"
    and cropped, centered, and squared so that both width and
    height have lengths "size". The frame must be transposed from
    Height-Width-Channels (HWC) to Channels-Height-Width (CHW).

    :param frame: input frame
    :param size: input size to encoder model
    :returns: resized and cropped frame
    """
    # Adaptative resize
    preprocessed = adaptive_resize(frame, size)
    # Center_crop
    (preprocessed, roi) = center_crop(preprocessed)
    # Transpose frame HWC -> CHW
    preprocessed = preprocessed.transpose((2, 0, 1))[None,]  # HWC -> CHW
    return preprocessed, roi


def encoder(
    preprocessed: np.ndarray,
    compiled_model: CompiledModel
) -> List:
    """
    Encoder Inference per frame. This function calls the network previously
    configured for the encoder model (compiled_model), extracts the data
    from the output node, and appends it in an array to be used by the decoder.

    :param: preprocessed: preprocessing frame
    :param: compiled_model: Encoder model network
    :returns: encoder_output: embedding layer that is appended with each arriving frame
    """
    output_key_en = compiled_model.output(0)

    # Get results on action-recognition-0001-encoder model
    infer_result_encoder = compiled_model([preprocessed])[output_key_en]
    return infer_result_encoder


def decoder(encoder_output: List, compiled_model_de: CompiledModel, labels) -> List:
    """
    Decoder inference per set of frames. This function concatenates the embedding layer
    froms the encoder output, transpose the array to match with the decoder input size.
    Calls the network previously configured for the decoder model (compiled_model_de), extracts
    the logits and normalize those to get confidence values along specified axis.
    Decodes top probabilities into corresponding label names

    :param: encoder_output: embedding layer for 16 frames
    :param: compiled_model_de: Decoder model network
    :returns: decoded_labels: The k most probable actions from the labels list
            decoded_top_probs: confidence for the k most probable actions
    """
    # Concatenate sample_duration frames in just one array
    decoder_input = np.concatenate(encoder_output, axis=0)
    # Organize input shape vector to the Decoder (shape: [1x16x512]]
    decoder_input = decoder_input.transpose((2, 0, 1, 3))
    decoder_input = np.squeeze(decoder_input, axis=3)
    output_key_de = compiled_model_de.output(0)
    # Get results on action-recognition-0001-decoder model
    result_de = compiled_model_de([decoder_input])[output_key_de]
    # Normalize logits to get confidence values along specified axis
    probs = softmax(result_de - np.max(result_de))
    # Decodes top probabilities into corresponding label names
    decoded_labels, decoded_top_probs = decode_output(probs, labels, top_k=3)
    return decoded_labels, decoded_top_probs


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Normalizes logits to get confidence values along specified axis
    x: np.array, axis=None
    """
    exp = np.exp(x)
    return exp / np.sum(exp, axis=None)
