from typing import Deque, List, Tuple
import logging
from classes_enum import ClassesEnum

logger = logging.getLogger()


def get_overlapping_clusters_across_time(
    clusters_info: Deque[List[Tuple[str, float, List]]],
    drowning_cluster: Tuple[str, float, List],
    index_batch: int,
    threshold_low: float,
    max_width: int,
    max_height: int,
    iou_thres: float = 0.1,
) -> List[Tuple[str, float, List]]:
    """
    Find overlapping drowning clusters with a drowning cluster across all batches in clusters_info,
    including the drowning_cluster itself and selecting the max confidence per batch from other batches,
    considering only ClassesEnum.drowning clusters.

    Args:
        clusters_info (Deque[List[Tuple[str, float, List]]]): Deque of lists, where each list contains tuples
                                                             (class_name, confidence, bbox) for all clusters in a batch,
                                                             with bbox as [x1, y1, x2, y2]. Holds the last N batches.
        drowning_cluster (Tuple[str, float, List]): Tuple (class_name, confidence, bbox) of the drowning cluster
                                                   with confidence > threshold_high, where bbox is [x1, y1, x2, y2].
        index_batch (int): Index of the batch in clusters_info where the drowning_cluster was found.
        threshold_low (float): Confidence threshold (ensures minimum confidence for ClassesEnum.drowning clusters in other batches).
        iou_thres (float): Intersection over Union (IoU) threshold for determining overlapping clusters (default 0.1).

    Returns:
        List[Tuple[str, float, List]]: List of tuples (class_name, confidence, bbox) including the drowning_cluster
                                      and the highest-confidence ClassesEnum.drowning overlapping cluster from each other batch,
                                      only including ClassesEnum.drowning clusters with confidence >= threshold_low.

    Raises:
        ValueError: If threshold_low or iou_thres is invalid (e.g., <0 or >1) or index_batch is out of range.
    """
    # Input validation
    if not 0 <= threshold_low <= 1:
        raise ValueError("threshold_low must be between 0 and 1.")

    if not 0 <= iou_thres <= 1:
        raise ValueError("iou_thres must be between 0 and 1.")

    if not clusters_info or index_batch < 0 or index_batch >= len(clusters_info):
        raise ValueError("Invalid index_batch or empty clusters_info.")

    # Extract the drowning cluster's bbox and confidence
    _, drowning_confidence, drowning_bbox = drowning_cluster
    if drowning_bbox is None or len(drowning_bbox) != 4:
        raise ValueError("Invalid drowning_cluster bbox format.")

    # Increase drowning_cluster bbox size by 20% for overlap matching (adjustable parameter)
    def expand_bbox(bbox, max_width, max_height, scale_factor=1.2):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        # Expand by 20% in each direction
        new_width = width * scale_factor
        new_height = height * scale_factor
        new_x1 = max(0, int(x1 - (new_width - width) / 2))
        new_y1 = max(0, int(y1 - (new_height - height) / 2))
        new_x2 = min(max_width, int(new_x1 + new_width))
        new_y2 = min(max_height, int(new_y1 + new_height))
        return [new_x1, new_y1, new_x2, new_y2]  # Return as list to match input format

    expanded_drowning_bbox = expand_bbox(
        bbox=drowning_bbox, max_width=max_width, max_height=max_height
    )

    overlapping_clusters = [drowning_cluster]  # Include the drowning_cluster directly
    processed_batches = (
        set()
    )  # Track batches already processed to ensure max 1 cluster per batch

    # Iterate over all batches except the one at index_batch
    for batch_idx, batch_results in enumerate(clusters_info):
        if batch_idx == index_batch:
            continue

        # Track the best (highest confidence) overlapping ClassesEnum.drowning cluster for this batch
        best_confidence = -1.0
        best_cluster = None

        # Check each cluster in the batch for overlap with the drowning_cluster, only considering ClassesEnum.drowning clusters
        for cluster in batch_results:
            class_name, confidence, bbox = cluster
            if class_name != ClassesEnum.drowning:  # Skip normal clusters
                continue
            if bbox is None or len(bbox) != 4:
                continue

            expanded_cluster_bbox = expand_bbox(
                bbox, max_width=max_width, max_height=max_height
            )
            # Check for overlap using intersection over union (IoU) threshold
            d_x1, d_y1, d_x2, d_y2 = expanded_drowning_bbox
            c_x1, c_y1, c_x2, c_y2 = expanded_cluster_bbox

            intersection = max(0, min(d_x2, c_x2) - max(d_x1, c_x1)) * max(
                0, min(d_y2, c_y2) - max(d_y1, c_y1)
            )
            union = (
                (d_x2 - d_x1) * (d_y2 - d_y1)
                + (c_x2 - c_x1) * (c_y2 - c_y1)
                - intersection
            )
            iou = intersection / union if union > 0 else 0

            if iou > iou_thres:  # Use the provided iou_thres
                # Update best_cluster if this has higher confidence and meets threshold_low
                if confidence > best_confidence and confidence >= threshold_low:
                    best_confidence = confidence
                    best_cluster = (ClassesEnum.drowning, confidence, bbox)

        # Add the best cluster for this batch if found
        if best_cluster and batch_idx not in processed_batches:
            overlapping_clusters.append(best_cluster)
            processed_batches.add(batch_idx)

    return overlapping_clusters


def contains_high_conf_drowning(clusters_info, threshold_high):
    for batch_idx, batch_results in enumerate(clusters_info):
        for cluster_idx, (class_name, confidence, bbox) in enumerate(batch_results):
            if class_name == ClassesEnum.drowning and confidence >= threshold_high:
                return True
    return False


def get_multi_segment_confirmation(
    clusters_info: Deque[List[Tuple[str, float, List]]],
    max_width: int,
    max_height: int,
    threshold_high: float = 0.8,
    threshold_low: float = 0.5,
    threshold_drowning_signal=0.75,
) -> Tuple[str, float]:
    """
    Determine the final class and confidence based on a history of cluster predictions, finding drowning clusters
    with confidence > threshold_high, checking overlapping drowning clusters across time, and requiring >=75% drowning
    clusters in any overlapping set, selecting the max confidence per batch, logging only the clusters from the
    group with the highest confidence among qualifying groups.

    Args:
        clusters_info (Deque[List[Tuple[str, float, List]]]): Deque of lists, where each list contains tuples
                                                             (class_name, confidence, bbox) for all clusters in a batch,
                                                             with bbox as [x1, y1, x2, y2]. The deque holds the last N
                                                             batches of cluster results.
        threshold_high (float): Confidence threshold to initially consider a cluster as ClassesEnum.drowning (default 0.8).
        threshold_low (float): Confidence threshold for minimum ClassesEnum.drowning confidence in overlapping clusters.

    Returns:
        tuple: (final_class, final_confidence) where final_class is ClassesEnum.drowning or ClassesEnum.normal,
               and final_confidence is the highest confidence among overlapping ClassesEnum.drowning clusters
               in the qualifying group with the maximum confidence (between 0 and 1) or 1.0 for ClassesEnum.normal.

    Raises:
        ValueError: If thresholds are invalid (e.g., <0 or >1).
    """
    # Input validation
    if not 0 <= threshold_high <= 1 or not 0 <= threshold_low <= 1:
        raise ValueError("Thresholds must be between 0 and 1.")

    if not clusters_info:
        return ClassesEnum.normal, 1.0, []

    total_batches = len(clusters_info)

    has_high_conf_drowning = contains_high_conf_drowning(clusters_info, threshold_high)
    if not has_high_conf_drowning:
        return (
            ClassesEnum.normal,
            1.0,
            [],
        )  # No high-confidence drowning clusters, classify as normal with full confidence

    drowning_candidates = []

    # if we found at least one drowning with high confidence threshold we can allow now with low threshold
    for batch_idx, batch_results in enumerate(clusters_info):
        for cluster_idx, (class_name, confidence, bbox) in enumerate(batch_results):
            if class_name == ClassesEnum.drowning and confidence >= threshold_low:
                drowning_candidates.append((class_name, confidence, bbox, batch_idx))

    # Process each drowning canznacdidate to find overlapping drowning clusters across time
    max_drowning_confidence = 0.0
    has_drowning_group = False
    best_drowning_group = []  # Store the group with the highest confidence

    for drowning_cluster, index_batch in [
        (cand[:3], cand[3]) for cand in drowning_candidates
    ]:
        # Get overlapping drowning clusters from other batches, including the drowning_cluster itself
        overlapping_clusters = get_overlapping_clusters_across_time(
            clusters_info=clusters_info,
            drowning_cluster=drowning_cluster,
            index_batch=index_batch,
            threshold_low=threshold_low,
            max_width=max_width,
            max_height=max_height,
        )

        # Count drowning clusters in the overlapping set (all are ClassesEnum.drowning in this case)
        num_clusters = len(overlapping_clusters)
        if num_clusters == 0:
            continue
        highest_conf_cluster = max(overlapping_clusters, key=lambda x: x[1])[1]
        # Require >=75% (0.75) of total batches to be drowning (i.e., num_clusters / total_batches >= 0.75)
        num_drowning = (
            num_clusters  # Since only ClassesEnum.drowning clusters are returned
        )
        if (
            num_drowning / total_batches >= threshold_drowning_signal
            and highest_conf_cluster > threshold_high
        ):
            has_drowning_group = True
            # Find the highest confidence among overlapping drowning clusters
            drowning_confidences = [
                confidence for _, confidence, _ in overlapping_clusters
            ]
            if drowning_confidences:
                current_max_confidence = max(drowning_confidences)
                if current_max_confidence > max_drowning_confidence:
                    max_drowning_confidence = current_max_confidence
                    # Store the current overlapping_clusters as the best group
                    best_drowning_group = sorted(
                        overlapping_clusters, key=lambda x: x[1], reverse=True
                    )

    # Determine final classification and confidence
    if has_drowning_group:
        final_class = ClassesEnum.drowning
        final_confidence = (
            max_drowning_confidence if max_drowning_confidence > 0 else 0.5
        )  # Default to 0.5 if no valid confidence
    else:
        final_class = ClassesEnum.normal
        final_confidence = 1.0

    final_confidence = min(
        max(final_confidence, 0.0), 1.0
    )  # Ensure confidence is between 0 and 1
    final_bbox = None
    # Log only the clusters from the group with the highest confidence, if it exists
    if best_drowning_group:
        final_bbox = max(best_drowning_group, key=lambda x: x[1])[2]
        for _, confidence, bbox in best_drowning_group:
            logger.info(
                f"Drowning group cluster (max confidence) - Confidence: {confidence:.4f}, BBox: {bbox}"
            )

    logger.info(f"Final classification - {final_class}: {final_confidence:.4f}")

    return final_class, final_confidence, final_bbox
