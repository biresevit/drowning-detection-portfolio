import numpy as np
import cv2
from classes_enum import ClassesEnum
import logging

logger = logging.getLogger()


def adjust_cols_rows_human_detection(
    video_width,
    video_height,
    input_cols,
    input_rows,
    min_tile_size=400,
    threshold_size_video=1200,
):
    if video_width < threshold_size_video and video_height < threshold_size_video:
        return 1, 1

    min_cols = int(video_width / min_tile_size)
    min_rows = int(video_height / min_tile_size)
    cols = max(1, min(min_cols, input_cols))
    rows = max(1, min(min_rows, input_rows))
    return cols, rows


def adjust_cols_rows(
    video_width, video_height, input_cols, input_rows, min_tile_size=250
):
    min_cols = int(video_width / min_tile_size)
    min_rows = int(video_height / min_tile_size)
    cols = max(1, min(min_cols, input_cols))
    rows = max(1, min(min_rows, input_rows))
    return cols, rows


def get_adjusted_size(width, height, max_width=1920, max_height=1080):
    # Check if resizing is needed
    if width <= max_width and height <= max_height:
        return width, height  # No need to resize, return original frame size

    # Calculate scaling factor to maintain aspect ratio
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale_factor = min(width_ratio, height_ratio)

    # Compute new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return new_width, new_height


def resize_big_frame(frame, max_width=1920, max_height=1080):
    """
    Resize a frame to fit within a maximum width and height while maintaining aspect ratio.

    Args:
        frame (numpy.ndarray): The input image/frame as a NumPy array.
        max_width (int): The maximum allowed width of the resized frame. Default is 1920.
        max_height (int): The maximum allowed height of the resized frame. Default is 1080.

    Returns:
        numpy.ndarray: The resized frame if dimensions exceed limits, otherwise the original frame.
    """
    # Get original frame dimensions
    height, width = frame.shape[:2]
    new_width, new_height = get_adjusted_size(width, height, max_width, max_height)
    # Resize the frame
    if new_width != width or new_height != height:
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        return frame
