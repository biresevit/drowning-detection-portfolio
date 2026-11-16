def normalize_bbox_corners(bbox, image_width, image_height):
    """
    Normalize bounding box coordinates [x1, y1, x2, y2] to range [0, 1].

    Args:
        bbox (list or tuple): [x1, y1, x2, y2] absolute coordinates.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        list: Normalized bbox [x1_norm, y1_norm, x2_norm, y2_norm].
    """
    x1, y1, x2, y2 = bbox
    return [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
