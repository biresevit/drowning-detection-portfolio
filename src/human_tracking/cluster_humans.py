from sklearn.cluster import KMeans
import numpy as np
import logging

logger = logging.getLogger()


class ClusterHumans:
    """
    A class to cluster human bounding boxes in video frames.
    """

    def __init__(self, num_total_crops, tracked_objects=None):
        """
        Initializes the ClusterHumans instance.

        Args:
            num_total_crops (int): Total number of crops.
            tracked_objects (dict, optional): Dictionary to store tracked objects. Defaults to None.
        """
        self.num_total_crops = num_total_crops
        self.tracked_objects = tracked_objects if tracked_objects else {}

    def calculate_union_area(self, cluster):
        """
        Calculate the union area of bounding boxes in a cluster.

        Args:
            cluster (list): List of bounding boxes in the cluster.

        Returns:
            float: The union area of the bounding boxes.

        Raises:
            Exception: If an error occurs during the calculation, it is logged.
        """
        try:
            x_min = min(bbox[0] for bbox in cluster)
            y_min = min(bbox[1] for bbox in cluster)
            x_max = max(bbox[2] for bbox in cluster)
            y_max = max(bbox[3] for bbox in cluster)
            return (x_max - x_min) * (y_max - y_min)
        except Exception as e:
            logger.error(f"Exception in calculate_union_area: {e}")
            return 0

    def kmeans_cluster(self, bboxes, num_clusters):
        """
        Cluster bounding boxes into a specified number of clusters using KMeans.

        Args:
            bboxes (list): List of bounding boxes to cluster.
            num_clusters (int): Number of clusters.

        Returns:
            dict: A dictionary where keys are cluster labels and values are lists of bounding boxes.

        Raises:
            Exception: If an error occurs during clustering, it is logged.
        """
        try:
            centroids = np.array(
                [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in bboxes]
            )
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(centroids)
            labels = kmeans.labels_

            clusters = {i: [] for i in range(num_clusters)}
            for bbox, label in zip(bboxes, labels):
                clusters[label].append(bbox)

            return clusters
        except Exception as e:
            logger.error(f"Exception in kmeans_cluster: {e}")
            return {}

    def generate_crops(self, bboxes):
        """
        Generate final crops by clustering bounding boxes and calculating union areas.

        Args:
            bboxes (list): List of bounding boxes to process.

        Returns:
            dict: A dictionary where keys are crop IDs and values are dictionaries with bounding boxes and frame counts.

        Raises:
            Exception: If an error occurs during crop generation, it is logged.
        """
        try:
            final_crops = {}
            if bboxes is None or len(bboxes) == 0:
                return final_crops
            num_clusters = min(len(bboxes), self.num_total_crops)
            clusters = self.kmeans_cluster(bboxes, num_clusters)

            crop_id = 1
            for cluster_bboxes in clusters.values():
                if len(cluster_bboxes) == 0:
                    logger.warning("Number of bbox is 0 in cluster_bboxes")
                    continue
                x_min = min(bbox[0] for bbox in cluster_bboxes)
                y_min = min(bbox[1] for bbox in cluster_bboxes)
                x_max = max(bbox[2] for bbox in cluster_bboxes)
                y_max = max(bbox[3] for bbox in cluster_bboxes)

                final_crops[crop_id] = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "frames_count": sum(
                        obj["frames_count"]
                        for obj in self.tracked_objects.values()
                        if obj["bbox"] in cluster_bboxes
                    ),
                }
                crop_id += 1

            return final_crops
        except Exception as e:
            logger.error(f"Exception in generate_crops: {e}")
            return {}

    def process(self, bboxes=None, candidate_ids=None):
        """
        Process bounding boxes and return final crops.

        When ``candidate_ids`` is provided, only those tracks are forwarded to
        generate_crops, focusing each cluster crop on suspicious persons.
        Falls back to all tracked objects if the candidate set is empty.

        Args:
            bboxes (list, optional): Explicit list of bboxes to cluster.
            candidate_ids (set, optional): Track IDs identified as drowning
                candidates by BehaviorAnalyses.

        Returns:
            dict: Crop ID → {bbox, frames_count}.
        """
        try:
            if bboxes is None and self.tracked_objects:
                if candidate_ids is not None:
                    bboxes = [
                        obj["bbox"]
                        for track_id, obj in self.tracked_objects.items()
                        if track_id in candidate_ids
                    ]
                else:
                    bboxes = [obj["bbox"] for obj in self.tracked_objects.values()]

            crops = self.generate_crops(bboxes)
            return crops
        except Exception as e:
            logger.error(f"Exception in process: {e}")
            return {}
