import logging
import os
logger = logging.getLogger()


def initialize_environment(use_cloud):
    """Initialize cloud environment variables and validate them."""
    azure_original_video_url = None
    azure_resized_video_url = None
    if use_cloud:
        azure_original_video_url = os.getenv("AZURE_FUNCTION_SAVE_VIDEO_URL")
        azure_resized_video_url = os.getenv("AZURE_FUNCTION_SAVE_RESIZED_VIDEO_URL")
        if azure_original_video_url is None or azure_resized_video_url is None:
            logger.error(
                "AZURE_FUNCTION_SAVE_VIDEO_URL or AZURE_RESIZED_VIDEO_URL is not defined!"
            )
            raise ValueError(
                "AZURE_FUNCTION_SAVE_VIDEO_URL or AZURE_RESIZED_VIDEO_URL is not defined!"
            )
    return azure_original_video_url, azure_resized_video_url

