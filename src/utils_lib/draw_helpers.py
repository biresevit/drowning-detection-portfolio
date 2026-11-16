
import queue
import time
import cv2
import logging

from classes_enum import ClassesEnum
import uuid
import os

logger = logging.getLogger()

def draw_alert_notification(
    frame, predicted_class, confidence, bbox, human_tracker, display_water_mask_state
):
    color = (0, 0, 255)
    top_left = (bbox[0], bbox[1])
    bottom_right = (bbox[2], bbox[3])
    cv2.rectangle(frame, top_left, bottom_right, color, 10)
    confidence *= 100
    cv2.putText(
        frame,
        f"{predicted_class.name} ({confidence:.2f}%)",
        (top_left[0] + 10, bottom_right[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    if display_water_mask_state:
        frame = human_tracker.apply_transparent_water_mask(frame)
    return frame

def can_save_frames(last_save, total_seconds):
    return last_save is None or time.perf_counter() - last_save > total_seconds

def draw_classification_result(
    info_predictions, frame, conf_high_thres, display_only_drowning
):
    try:
        # Collect all predictions
        drowning_preds = []
        normal_preds = []

        for idx, (class_name, confidence, cluster_bbox) in enumerate(info_predictions):
            if class_name == ClassesEnum.drowning:
                drowning_preds.append((class_name, confidence, cluster_bbox))
            elif class_name == ClassesEnum.normal and not display_only_drowning:
                normal_preds.append((class_name, confidence, cluster_bbox))

        # Function to calculate if two boxes overlap
        def boxes_overlap(box1, box2):
            x1, y1, x2, y2 = box1
            x3, y3, x4, y4 = box2
            return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

        # Process drowning predictions
        if drowning_preds:
            if display_only_drowning:
                # Initial grouping of overlapping boxes
                groups = []
                used = set()  # Track indices that have been grouped

                for i, (_, conf_i, bbox_i) in enumerate(drowning_preds):
                    if i in used:
                        continue

                    current_group = [(conf_i, bbox_i)]
                    used.add(i)

                    # Check against all unused elements
                    for j, (_, conf_j, bbox_j) in enumerate(drowning_preds):
                        if j in used or i == j:
                            continue
                        if boxes_overlap(bbox_i, bbox_j):
                            current_group.append((conf_j, bbox_j))
                            used.add(j)

                    # Check against all existing groups
                    for group in groups:
                        if any(
                            boxes_overlap(bbox_i, group_bbox) for _, group_bbox in group
                        ):
                            group.extend(current_group)
                            current_group = (
                                []
                            )  # Clear current_group to avoid adding it again
                            break

                    # If current_group is not empty, add it as a new group
                    if current_group:
                        groups.append(current_group)

                # Merge groups that are connected
                merged = True
                while merged:
                    merged = False
                    i = 0
                    while i < len(groups) - 1:
                        j = i + 1
                        while j < len(groups):
                            if any(
                                boxes_overlap(bbox1, bbox2)
                                for _, bbox1 in groups[i]
                                for _, bbox2 in groups[j]
                            ):
                                groups[i].extend(groups[j])
                                groups.pop(j)
                                merged = True
                            else:
                                j += 1
                        i += 1

                # Draw the highest confidence box from each group
                for group in groups:
                    best_conf, best_bbox = max(
                        group, key=lambda x: x[0]
                    )  # Max by confidence
                    color = (
                        (0, 0, 255) if best_conf >= conf_high_thres else (0, 165, 255)
                    )
                    top_left = (best_bbox[0], best_bbox[1])
                    bottom_right = (best_bbox[2], best_bbox[3])
                    cv2.rectangle(frame, top_left, bottom_right, color, 3)
                    best_conf *= 100
                    text_position = (
                        top_left[0] + 10,
                        bottom_right[1] - 10,
                    )  # Lower left corner
                    cv2.putText(
                        frame,
                        f"drowning ({best_conf:.2f}%)",
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
            else:
                # Draw all drowning predictions without grouping
                for class_name, confidence, cluster_bbox in drowning_preds:
                    color = (
                        (0, 0, 255) if confidence >= conf_high_thres else (0, 165, 255)
                    )
                    top_left = (cluster_bbox[0], cluster_bbox[1])
                    bottom_right = (cluster_bbox[2], cluster_bbox[3])
                    cv2.rectangle(frame, top_left, bottom_right, color, 3)
                    confidence *= 100
                    text_position = (
                        top_left[0] + 10,
                        bottom_right[1] - 10,
                    )  # Lower left corner
                    cv2.putText(
                        frame,
                        f"{class_name.name} ({confidence:.2f}%)",
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        # Draw normal predictions if display_only_drowning is False
        if not display_only_drowning:
            for class_name, confidence, cluster_bbox in normal_preds:
                color = (0, 255, 0)
                top_left = (cluster_bbox[0], cluster_bbox[1])
                bottom_right = (cluster_bbox[2], cluster_bbox[3])
                cv2.rectangle(frame, top_left, bottom_right, color, 3)
                confidence *= 100
                cv2.putText(
                    frame,
                    f"{class_name.name} ({confidence:.2f}%)",
                    (top_left[0] + 10, top_left[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    except Exception as e:
        logger.error(f"Error during draw_classification_result: {e}")


def draw_outer_frame(image, predicted_class, thickness=10):
    if len(image.shape) != 3 or image.shape[2] not in [3, 4]:
        raise ValueError(
            f"Image must have 3 (BGR) or 4 (RGBA) channels, got shape {image.shape}"
        )
    color = (0, 0, 255) if predicted_class == ClassesEnum.drowning else (0, 255, 0)
    height, width = image.shape[:2]
    output_image = image.copy()
    if image.shape[2] == 3:
        output_image = cv2.rectangle(
            output_image, (0, 0), (width, height), color, thickness
        )
    elif image.shape[2] == 4:
        b, g, r, a = cv2.split(output_image)
        bgr_image = cv2.merge([b, g, r])
        bgr_image = cv2.rectangle(bgr_image, (0, 0), (width, height), color, thickness)
        b, g, r = cv2.split(bgr_image)
        output_image = cv2.merge([b, g, r, a])
    return output_image


def display_last_frame(
    last_frame,
    processing_done_event,
    shared_data_queue,
    display_human_detections_state,
    conf_high_thres,
    display_only_drowning_state,
    human_tracker,
    display_water_mask_state,
    frame_display_queue,
    info_predictions,
    predicted_class,
):

    if last_frame is not None:
        while not processing_done_event.is_set():
            try:
                try:
                    last_predicted_class, _, _, last_info_predictions = (
                        shared_data_queue.get_nowait()
                    )
                    if (
                        last_info_predictions is not None
                        and last_predicted_class is not None
                    ):
                        info_predictions = last_info_predictions
                        predicted_class = last_predicted_class
                except queue.Empty:
                    pass
                frame = last_frame.copy()
                    
                draw_results(
                    frame,
                    info_predictions,
                    conf_high_thres,
                    display_only_drowning_state.is_set(),
                    display_water_mask_state.is_set(),
                    display_human_detections_state.is_set(),
                    human_tracker,
                    predicted_class,
                    frame_display_queue,
                )

                draw_classification_result(
                    info_predictions,
                    frame,
                    conf_high_thres,
                    display_only_drowning_state.is_set(),
                )
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error during post-processing display: {e}")
                break


def draw_results(
    frame,
    info_predictions,
    conf_high_thres,
    display_only_drowning_state,
    display_water_mask_state,
    display_human_detections_state,
    human_tracker,
    predicted_class,
    frame_display_queue,
):
    try:
        if info_predictions and len(info_predictions) > 0:
            draw_classification_result(
                info_predictions, frame, conf_high_thres, display_only_drowning_state
            )

        if display_human_detections_state:
            human_tracker.draw_object_detections(frame)
        if display_water_mask_state:
            frame = human_tracker.apply_transparent_water_mask(frame)
        frame = draw_outer_frame(frame, predicted_class)
        if frame_display_queue is not None:
            try:
                frame_display_queue.put_nowait(frame)
            except queue.Full:
                pass
    except Exception as e:
        logger.error(f"Error during draw_results: {e}")
    return frame





