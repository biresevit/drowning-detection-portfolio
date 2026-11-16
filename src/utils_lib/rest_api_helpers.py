import logging
from utils_lib.bbox_helpers import normalize_bbox_corners

logger = logging.getLogger()
    
def normalize_info_predictions(info_predictions, image_width, image_height):
    """
    Normalize bbox at index 2 and round confidence at index 1 
    for each item in the info_predictions list.
    """
    if not info_predictions:
        return []

    normalized = []
    for item in info_predictions:
        try:
            # Convert tuple to list so we can modify it
            item = list(item)

            # Round confidence (index 1) if it's a float
            if len(item) > 1 and isinstance(item[1], float):
                item[1] = round(item[1], 2)

            # Normalize bbox (index 2) if it's a 4-element list/tuple
            if len(item) > 2 and isinstance(item[2], (list, tuple)) and len(item[2]) == 4:
                item[2] = normalize_bbox_corners(item[2], image_width, image_height)
            else:
                logger.warning(f"Skipping bbox normalization due to invalid bbox format: {item}")

            normalized.append(tuple(item))

        except Exception as e:
            # If anything goes wrong, log it and skip this item
            logger.exception(f"Failed to normalize info_prediction item {item}: {e}")
            continue  # explicitly skip the problematic item

    return normalized


def convert_for_json(obj):
    """Recursively convert objects to JSON-serializable formats."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(x) for x in obj]
    elif hasattr(obj, "name"):  # Enum type
        return obj.name
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)
