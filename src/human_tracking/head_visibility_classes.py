"""
head_visibility_classes.py

Single source of truth for head-visibility class names used across
training, inference, annotation UI, and behavior analysis.

False Positive is annotation-only: images labelled FP are excluded from
the training split. The trained model therefore only outputs the three
head-visibility classes.
"""

from enum import Enum


class HeadVisibilityClass(str, Enum):
    """String enum — values equal the folder / class names used on disk.

    __str__ returns the plain value so str(member) == member.value,
    which is required for Path joining, logging, and Tkinter StringVar.
    (Python 3.11 changed the default __str__ for mixed-in enums to
    return 'ClassName.MEMBER' instead of the value.)
    """
    HEAD_AND_SHOULDERS_VISIBLE     = "head_and_shoulders_visible"
    HEAD_CLEARLY_VISIBLE           = "head_clearly_visible"
    HEAD_HORIZONTAL_OR_UNDER_WATER = "head_horizontal_or_under_water"
    FALSE_POSITIVE                 = "false_positive"

    def __str__(self) -> str:
        return self.value


# Classes included in training and model inference (FP is annotation-only).
TRAINING_CLASSES: list[HeadVisibilityClass] = [
    HeadVisibilityClass.HEAD_AND_SHOULDERS_VISIBLE,
    HeadVisibilityClass.HEAD_CLEARLY_VISIBLE,
    HeadVisibilityClass.HEAD_HORIZONTAL_OR_UNDER_WATER,
]

# Alphabetical order produced by torchvision ImageFolder — matches model output indices.
INFERENCE_CLASS_NAMES: list[str] = sorted(c.value for c in TRAINING_CLASSES)

# Shared colour per class, so the review UI and the annotated video agree.
CLASS_COLORS_HEX: dict[str, str] = {
    HeadVisibilityClass.HEAD_AND_SHOULDERS_VISIBLE.value:     "#388e3c",   # green
    HeadVisibilityClass.HEAD_CLEARLY_VISIBLE.value:           "#1976d2",   # blue
    HeadVisibilityClass.HEAD_HORIZONTAL_OR_UNDER_WATER.value: "#c62828",   # red
    HeadVisibilityClass.FALSE_POSITIVE.value:                 "#607d8b",   # grey
}

# Short labels for drawing on frames, where there is no room for the full name.
CLASS_SHORT_NAMES: dict[str, str] = {
    HeadVisibilityClass.HEAD_AND_SHOULDERS_VISIBLE.value:     "h&s",
    HeadVisibilityClass.HEAD_CLEARLY_VISIBLE.value:           "clear",
    HeadVisibilityClass.HEAD_HORIZONTAL_OR_UNDER_WATER.value: "under",
    HeadVisibilityClass.FALSE_POSITIVE.value:                 "fp",
}


# Stored per frame when a tracked person was NOT detected. It counts toward the
# observation window but never toward "head visible" evidence — a missed detection in
# a water zone often means submerged, which is exactly the evidence that would
# otherwise vanish from the denominator.
NOT_DETECTED = "not_detected"


def class_color_bgr(class_name: str, default=(200, 200, 200)) -> tuple[int, int, int]:
    """Class colour as an OpenCV BGR tuple (cv2 is BGR, the hex above is RGB)."""
    hex_color = CLASS_COLORS_HEX.get(class_name)
    if not hex_color:
        return default
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
