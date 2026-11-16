import threading
from ultralytics import YOLO
import time
from collections import deque
import logging

logger = logging.getLogger()


class ObjectTracker:
    def __init__(self, human_model_path, conf, iou, classes, deque_length, imgsz):
        """
        Initializes the ObjectTracker instance.

        Args:
            human_model_path (str): Path to the YOLO model for human detection.
            tile_cols (int): Number of columns to divide the frame into tiles.
            tile_rows (int): Number of rows to divide the frame into tiles.
            conf (float): Confidence threshold for object detection.
            iou (float): Intersection over Union (IoU) threshold for object detection.
            classes (list): List of classes to detect.
            deque_lenght (int): Maximum length of the deque to store tracking results per frame.
            imgsz (int) : size of the image to be processed by the model.

        Attributes:
            human_model_path (str): Path to the YOLO model for human detection.
            conf (float): Confidence threshold for object detection.
            iou (float): Intersection over Union (IoU) threshold for object detection.
            classes (list): List of classes to detect.
            model_lock (threading.Lock): Lock to ensure thread safety during model access.
            last_results (dict): Dictionary to store the last tracking results.
            tracking_per_frame (collections.deque): Deque to store tracking results for individual frames.
            models (list): List of YOLO model instances, one for each tile.
        """
        try:
            self.human_model_path = human_model_path
            self.conf = conf
            self.iou = iou
            self.classes = classes
            self.model_lock = threading.Lock()
            self.last_results = {}
            self.models = None
            self.frame_width = None
            self.frame_height = None
            self.imgsz = imgsz
            self.deque_lenght = deque_length
            self.tracking_per_frame = deque(maxlen=deque_length)
            self.print_counter = 0
        except Exception as e:
            logger.error(f"Error in __init__: {e}")

    def set_deque_lenght(self, deque_lenght):
        """
        Set the maximum length of the deque to store tracking results per frame.

        Args:
            deque_lenght (int): Maximum length of the deque to store tracking results per frame.
        """
        try:
            self.deque_lenght = deque_lenght
            self.tracking_per_frame = deque(maxlen=deque_lenght)
        except Exception as e:
            logger.error(f"Error in set_deque_lenght: {e}")

    def reset(self):
        """
        Reset the object tracker by clearing the tracking results.

        This method clears the tracking results stored in `self.tracking_per_frame`.
        """
        try:
            self.tracking_per_frame.clear()
        except Exception as e:
            logger.error(f"Error in reset: {e}")

    def initialize(self, frame_width, frame_height, tile_rows, tile_cols):
        """
        Set the number of tile rows and columns for the object tracker.

        This method initializes the number of tile rows and columns and pre-creates
        a list of models, one for each tile. If the number of models does not match
        the product of tile rows and columns, it reloads the models accordingly.

        Args:
            tile_rows (int): The number of tile rows.
            tile_cols (int): The number of tile columns.

        Raises:
            Exception: If there is an error during the setting of tile rows and columns.
        """
        try:
            self.tile_rows = tile_rows
            self.tile_cols = tile_cols
            self.frame_width = frame_width
            self.frame_height = frame_height
            self.reset()
            self.models = [
                YOLO(self.human_model_path)
                for _ in range(self.tile_cols * self.tile_rows)
            ]
            logger.info(
                f"Models are loaded for object tracker: {tile_cols} x {tile_rows}"
            )
        except Exception as e:
            logger.error(f"Error in initialize: {e}")

    def process_tile(
        self, frame, x_start, y_start, x_end, y_end, model, tile_id, frame_tracking
    ):
        """
        Process a single tile of the input frame by running object tracking.

        Args:
            frame (numpy.ndarray): The complete input frame from the video.
            x_start (int): The starting x-coordinate of the tile in the global frame.
            y_start (int): The starting y-coordinate of the tile in the global frame.
            x_end (int): The ending x-coordinate of the tile in the global frame.
            y_end (int): The ending y-coordinate of the tile in the global frame.
            model (YOLO): The YOLO model instance assigned to this tile for processing.
            tile_id (int): The unique ID representing this tile.

        Updates:
            - Updates self.last_results with results for the current frame.
            - Updates self.tracking_results by merging bounding boxes for the same unique ID.
        """
        try:
            self.frame_width = frame.shape[1]
            self.frame_height = frame.shape[0]
            # Extract the tile and adjust size
            tile = frame[y_start:y_end, x_start:x_end]
            # Perform tracking on the tile
            with self.model_lock:
                predictions = model.track(
                    tile,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.classes,
                    persist=True,
                    verbose=False,
                    tracker="custom_tracker.yaml",
                )

            for pred in predictions:
                for obj in pred.boxes:
                    x1, y1, x2, y2 = obj.xyxy[0].tolist()

                    # Adjust coordinates to global frame
                    global_x1 = max(0, min(x1 + x_start, self.frame_width))
                    global_y1 = max(0, min(y1 + y_start, self.frame_height))
                    global_x2 = max(0, min(x2 + x_start, self.frame_width))
                    global_y2 = max(0, min(y2 + y_start, self.frame_height))
                    original_id = int(obj.id.item()) if obj.id is not None else None

                    if original_id is None:
                        continue  # Skip objects with no ID

                    # Create a unique ID combining tile ID and object ID
                    if self.tile_rows == 1 and self.tile_cols == 1:
                        unique_id = f"{original_id}"
                    else:
                        unique_id = f"{tile_id}_{original_id}"

                    # Save local results for the current frame
                    frame_tracking[unique_id] = {
                        "id": unique_id,
                        "bbox": [global_x1, global_y1, global_x2, global_y2],
                        "confidence": obj.conf.item(),
                        "class": int(obj.cls.item()),
                        "frames_count": 1,
                    }

        except Exception as e:
            logger.error(f"Error in process_tile: {e}")

    def get_last_frame_tracking(self):
        """
        Returns the tracking results of the last processed frame.

        Returns:
            dict: The tracking results of the last processed frame if available, otherwise None.
        """
        try:
            if self.tracking_per_frame:
                return self.tracking_per_frame[-1]
            else:
                return None
        except Exception as e:
            logger.error(f"Error in get_last_frame_tracking: {e}")
            return None

    def merge_bboxes_same_object_id(self):
        """
        Merges bounding boxes for the same object ID across multiple frames.

        This function iterates over the `self.tracking_per_frame` deque, which contains
        tracking results for individual frames. For each unique object ID, it merges
        the bounding boxes to cover all positions the object has occupied across frames.
        The final merged bounding boxes are stored in `self.tracking_results`.

        The merging process involves:
        - Iterating over each frame's tracking results.
        - For each unique object ID, checking if it already exists in `self.tracking_results`.
        - If it exists, merging the bounding boxes to cover the minimum and maximum extents.
        - If it does not exist, adding the result to `self.tracking_results`.

        The function uses a lock (`self.lock`) to ensure thread safety during the update
        of `self.tracking_results`.

        Raises:
            Exception: If an error occurs during the merging process, it is logged.
        """
        try:
            tracking_results = {}
            for frame_tracking in self.tracking_per_frame:
                for unique_id, result in frame_tracking.items():
                    if unique_id in tracking_results:
                        # Merge bounding boxes to cover all positions
                        existing_bbox = tracking_results[unique_id]["bbox"]
                        merged_bbox = [
                            min(existing_bbox[0], result["bbox"][0]),  # x1
                            min(existing_bbox[1], result["bbox"][1]),  # y1
                            max(existing_bbox[2], result["bbox"][2]),  # x2
                            max(existing_bbox[3], result["bbox"][3]),  # y2
                        ]
                        # Ensure no negative values in bounding box
                        merged_bbox = [max(0, coord) for coord in merged_bbox]
                        tracking_results[unique_id]["bbox"] = merged_bbox
                        tracking_results[unique_id]["frames_count"] += 1
                    else:
                        # Ensure no negative values in bounding box
                        result["bbox"] = [max(0, coord) for coord in result["bbox"]]
                        tracking_results[unique_id] = result
            return tracking_results
        except Exception as e:
            logger.error(f"Error in merge_bboxes_same_object_id: {e}")
            return None

    def process_frame(self, frame):
        """
        Perform multi-threaded tracking on tiled frames.

        Args:
            frame (numpy.ndarray): Full input frame.

        Returns:
            dict: Merged tracking results for the entire frame.
        """
        try:
            height, width, _ = frame.shape
            self.last_results = {}
            # Calculate tile dimensions
            tile_width = width // self.tile_cols
            tile_height = height // self.tile_rows
            # Store results for the current tile (no merging here)
            frame_tracking = {}
            if self.tile_rows == 1 and self.tile_cols == 1:
                # Process the single tile directly without threading
                self.process_tile(
                    frame, 0, 0, width, height, self.models[0], 0, frame_tracking
                )
            else:
                # Create threads for each tile
                threads = []

                for row in range(self.tile_rows):
                    for col in range(self.tile_cols):
                        # Calculate tile boundaries
                        x_start = col * tile_width
                        y_start = row * tile_height
                        x_end = min(x_start + tile_width, width)
                        y_end = min(y_start + tile_height, height)

                        # Get the model for this tile
                        model_index = row * self.tile_cols + col
                        model = self.models[model_index]

                        # Create a unique tile identifier
                        tile_id = row * self.tile_cols + col

                        # Create a thread to process this tile
                        thread = threading.Thread(
                            target=self.process_tile,
                            args=(
                                frame,
                                x_start,
                                y_start,
                                x_end,
                                y_end,
                                model,
                                tile_id,
                                frame_tracking,
                            ),
                        )
                        threads.append(thread)
                        thread.start()

                # Wait for all threads to complete
                for id, thread in enumerate(threads):
                    thread.join()

            self.tracking_per_frame.append(frame_tracking)
        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
