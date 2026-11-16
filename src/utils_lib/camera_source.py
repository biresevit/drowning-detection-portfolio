def is_live_camera_source(video_source):
    """
    Determines if the provided source should be treated as a live camera.

    Args:
        video_source (str or None): Path to folder or stream URL.

    Returns:
        bool: True if live camera, False if local videos folder.
    """
    if not video_source:
        return True  # None or empty string → live camera

    videos_folder_str = str(video_source).lower()
    live_indicators = ["rtsp", "youtube", "googlevideo"]

    return any(indicator in videos_folder_str for indicator in live_indicators)
