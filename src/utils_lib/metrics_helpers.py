import logging
import queue
import threading
import time

from collect_device_metrics import MetricType
from collect_device_metrics import meter, log_metrics_thread

logger = logging.getLogger()

def initialize_metrics_thread(use_cloud, stop_event, mac_address):
    """Initialize the metrics thread if cloud is enabled."""
    metric_queue = queue.Queue()
    metric_thread = None
    if use_cloud and meter is not None:
        metric_thread = threading.Thread(
            target=log_metrics_thread,
            args=(metric_queue, stop_event, mac_address),
            daemon=True,
        )
        metric_thread.start()
    return metric_queue, metric_thread

def log_metrics(metric_queue, is_camera_working, last_upload_camera_working, last_upload_last_seen, metric_log_interval=60, last_seen_metric_log_interval=15):
    """Log camera and last seen metrics to the metric queue."""
    now = time.perf_counter()
    if last_upload_last_seen is None or (now - last_upload_last_seen) >= last_seen_metric_log_interval:
        try:
            metric_queue.put({"type": MetricType.LAST_SEEN, "last_seen": int(time.time())}, timeout=1)
            last_upload_last_seen = now
        except queue.Full:
            logger.warning("Metric queue full, skipping LAST_SEEN metric")
    if last_upload_camera_working is None or (now - last_upload_camera_working) >= metric_log_interval:
        try:
            metric_queue.put({"type": MetricType.CAMERA, "is_camera_working": is_camera_working}, timeout=1)
            last_upload_camera_working = now
        except queue.Full:
            logger.warning("Metric queue full, skipping CAMERA metric")
    return last_upload_camera_working, last_upload_last_seen


