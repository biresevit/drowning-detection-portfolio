import sys
import os

# Get the root directory of the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the project root to sys.path
sys.path.append(project_root)
from action_recognition.multi_segment_confirmation import get_multi_segment_confirmation
import time
from collections import deque
from classes_enum import ClassesEnum
from typing import Deque, List, Tuple
import queue
import logging
import threading
import cv2
logger = logging.getLogger()


def classification_thread(
    model,
    classification_queue,
    shared_data_queue,
    num_batches,
    conf_high_thres,
    conf_low_thres,
    stop_event,
    max_width,
    max_height,
):
    """Thread to process action classification and update shared_data_queue."""
    results_history = deque(maxlen=num_batches)
    clusters_info = deque(maxlen=num_batches)
    start_batch = time.perf_counter()

    while not stop_event.is_set():
        try:
            # Get frames_buffer and clusters (non-blocking with timeout)
            frames_buffer_copy, clusters = classification_queue.get(timeout=1)
            start = time.perf_counter()

            num_clusters = len(clusters)
            cluster_bboxes = get_bboxes_from_clusters(clusters)

            cluster_batches = []
            for frame in frames_buffer_copy:
                crops = get_crop_images(frame, cluster_bboxes)
                for idx, crop in enumerate(crops):
                    if len(cluster_batches) <= idx:
                        cluster_batches.append([])
                    cluster_batches[idx].append(crop)

            if num_clusters == 0:
                overall_class_name, overall_confidence = ClassesEnum.normal, 1.0
                clusters_results = []
            else:
                start_inference = time.perf_counter()
                logger.info(f"Running inference on {num_clusters} clusters...")
                clusters_results = model.predict_class(cluster_batches)
                logger.info(
                    f"Elapsed time for inference: {time.perf_counter() - start_inference:.3f}s"
                )

                for idx, cluster_res in enumerate(clusters_results):
                    class_name, confidence = cluster_res
                    clusters_results[idx] = (
                        get_class_enum(class_name),
                        round(confidence, 3),
                    )
                for idx, cluster_res in enumerate(clusters_results):
                    class_name, confidence = cluster_res
                    if (
                        class_name == ClassesEnum.normal
                        and (1 - confidence) > conf_low_thres
                    ):
                        clusters_results[idx] = (ClassesEnum.drowning, 1 - confidence)
                overall_class_name, overall_confidence = get_class_over_clusters(
                    clusters_results, conf_low_thres
                )
                logger.info(
                    f"Elapsed: {time.perf_counter() - start:.3f}s for clusters: {len(cluster_bboxes)} | {overall_class_name}: {overall_confidence:.2f}, size of queue: {classification_queue.qsize()}"
                )

            results_history.append((overall_class_name, overall_confidence))
            prediction_results = []
            cluster_aggregrated = []
            for idx, cluster_res in enumerate(clusters_results):
                class_name, conf = cluster_res
                bbox = cluster_bboxes[idx]
                cluster_aggregrated.append((class_name, conf, bbox))
            clusters_info.append(cluster_aggregrated)
                   
            # so that we can fire alert with high conf detection even if we don't collect full batch
            if len(results_history) == num_batches - 2:
                results_history.append((ClassesEnum.normal, 1.0))
                results_history.append((ClassesEnum.normal, 1.0))
                clusters_info.appendleft([])
                clusters_info.appendleft([])
                   
            if len(results_history) == num_batches:
                final_class, final_confidence, final_bbox = (
                    get_multi_segment_confirmation(
                        clusters_info=clusters_info,
                        threshold_high=conf_high_thres,
                        threshold_low=conf_low_thres,
                        threshold_drowning_signal=0.75,
                        max_width=max_width,
                        max_height=max_height,
                    )
                )

                if final_class == ClassesEnum.normal:
                    # here we use conf_high_thres for both threshold as we only want to count high confidence drowning
                    final_class, final_confidence, final_bbox = (
                        get_multi_segment_confirmation(
                            clusters_info=clusters_info,
                            threshold_high=conf_high_thres,
                            threshold_low=conf_high_thres,
                            threshold_drowning_signal=0.5,
                            max_width=max_width,
                            max_height=max_height,
                        )
                    )

                prediction_results = get_prediction_infos(
                    cluster_bboxes, clusters_results, conf_low_thres
                )
                update_shared_data_queue(
                    shared_data_queue,
                    (final_class, final_confidence, final_bbox, prediction_results),
                )
                logger.info(
                    f"Final classification - {final_class}: {final_confidence:.4f}, time: {time.perf_counter() - start_batch:.3f}s"
                )
                start_batch = time.perf_counter()
            else:
                prediction_results = get_prediction_infos(
                    cluster_bboxes, clusters_results, conf_low_thres
                )
                update_shared_data_queue(
                    shared_data_queue, (ClassesEnum.normal, 1.0, [], prediction_results)
                )

            classification_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in classification thread: {e}")
            classification_queue.task_done()


def processing_thread(
    frame_queue,
    model,
    human_tracker,
    num_frames,
    shared_data_queue,
    slide_window_perc,
    processing_done_event,
    stop_event,
    conf_high_thres,
    conf_low_thres,
    factor_tracking,
    num_batches,
    max_width,
    max_height,
):
    frames_buffer = deque(maxlen=num_frames)
    counter = 0

    # Initialize classification thread
    classification_queue = queue.Queue(maxsize=5)  # Limit queue size
    class_thread = threading.Thread(
        target=classification_thread,
        args=(
            model,
            classification_queue,
            shared_data_queue,
            num_batches,
            conf_high_thres,
            conf_low_thres,
            stop_event,
            max_width,
            max_height,
        ),
        daemon=True,
    )
    class_thread.start()
    time_batch = time.perf_counter()
    while not stop_event.is_set() or frame_queue.qsize() > 0:
        try:
            if frame_queue.qsize() > 5:
                logger.warning(
                    f"Number of elements in frame_queue: {frame_queue.qsize()}"
                )
            frame = frame_queue.get(timeout=1)
            start_process_tracker = time.perf_counter()
            human_tracker.process_frame(frame)
            #logger.info(f"Elapsed time for human tracker process_frame: {time.perf_counter() - start_process_tracker:.3f}s, frames in queue: {frame_queue.qsize()}")

            if counter % factor_tracking == 0:
                counter = 0
                # x3d requires rgb frames
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                
                frames_buffer.append(frame)
            counter += 1

            if len(frames_buffer) == num_frames:
                logger.info(
                    f"Processing batch, time since last batch: {time.perf_counter() - time_batch:.3f}s"
                )
                time_batch = time.perf_counter()
                human_tracker.compute_clusters()
                clusters = human_tracker.merged_clusters

                try:
                    classification_queue.put((list(frames_buffer), clusters), timeout=1)
                    slide_window_step = int(slide_window_perc * len(frames_buffer))
                    for _ in range(slide_window_step):
                        if frames_buffer:
                            frames_buffer.popleft()
                except queue.Full:
                    logger.warning("Classification queue full, dropping oldest batch")
                    classification_queue.get()
                    classification_queue.task_done()
                    classification_queue.put((list(frames_buffer), clusters))

        except queue.Empty:
            continue

    if processing_done_event:
        time.sleep(5)       
        logger.info("Processing thread completed, setting processing_done_event")
        processing_done_event.set()


def get_class_enum(class_name):
    if class_name == ClassesEnum.drowning.name:
        return ClassesEnum.drowning
    else:
        return ClassesEnum.normal


def get_bboxes_from_clusters(clusters):
    bboxes = []
    for obj in clusters.values():
        x_min, y_min, x_max, y_max = map(int, obj["bbox"])
        bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes


def get_crop_images(frame, bboxes):
    crops = []
    for obj in bboxes:
        x_min, y_min, x_max, y_max = map(int, obj)
        w = x_max - x_min
        h = y_max - y_min
        if w < 0 or h < 0:
            raise ValueError(f"Invalid bounding box: {obj}")
        crop = frame[y_min : y_min + h, x_min : x_min + w]
        crops.append(crop)
    return crops


def get_class_over_clusters(
    clusters_results: List[Tuple[str, float]], min_thres: float
) -> Tuple[str, float]:
    max_drowning_confidence = None
    total_confidence = 0
    count = 0
    for class_name, confidence in clusters_results:
        total_confidence += confidence
        count += 1
        if class_name == ClassesEnum.drowning and confidence >= min_thres:
            if max_drowning_confidence is None or confidence > max_drowning_confidence:
                max_drowning_confidence = confidence
        elif class_name == ClassesEnum.normal and 1 - confidence >= min_thres:
            if max_drowning_confidence is None or confidence > max_drowning_confidence:
                max_drowning_confidence = 1 - confidence
    if max_drowning_confidence is not None:
        return ClassesEnum.drowning, max_drowning_confidence
    else:
        avg_confidence = total_confidence / count if count > 0 else 0
        return ClassesEnum.normal, avg_confidence


def get_prediction_infos(cluster_bboxes, clusters_results, conf_low_thres):
    prediction_results = []
    for idx, cluster_bbox in enumerate(cluster_bboxes):
        class_name, confidence = clusters_results[idx]
        if class_name == ClassesEnum.normal.name and (1 - confidence) > conf_low_thres:
            class_name = ClassesEnum.drowning.name
            confidence = 1 - confidence
        prediction_results.append((class_name, confidence, cluster_bbox))
    return prediction_results


def update_shared_data_queue(shared_data_queue, prediction_data):
    try:
        shared_data_queue.put_nowait(prediction_data)
    except queue.Full:
        shared_data_queue.get_nowait()
        shared_data_queue.put_nowait(prediction_data)
