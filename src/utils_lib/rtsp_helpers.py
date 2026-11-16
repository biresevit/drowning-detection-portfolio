import os
from classes_enum import CAMERA_TYPE
from urllib.parse import quote


def get_rtsp_url(
    ip_address: str, cam_type: CAMERA_TYPE, username="", password=""
) -> str:
    username = "admin" if username == "" else username
    if password == "":
        # Credentials are supplied per installation, never stored in source.
        password = os.environ.get("CAMERA_PASSWORD", "")
    port = "554"

    if cam_type == CAMERA_TYPE.CTRONICS:
        stream_path = "11"  # Main stream for Ctronics CTIPC-640C
        return f"rtsp://{username}:{password}@{ip_address}:{port}/{stream_path}"

    elif cam_type == CAMERA_TYPE.REOLINK:
        stream_path = "Preview_01_main"
        return f"rtsp://{username}:{password}@{ip_address}:{port}/{stream_path}"

    elif cam_type == CAMERA_TYPE.DAHUA:
        channel = 1  # Adjust if using a different channel
        subtype = 0  # 0 = main stream, 1 = sub stream
        return f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"

    elif cam_type == CAMERA_TYPE.ZHEJIANG:
        password_encoded = quote(password, safe="")
        return f"rtsp://{username}:{password_encoded}@{ip_address}:{port}/stream1"

    elif cam_type == CAMERA_TYPE.TVT:
        return f"rtsp://{username}:{password}@{ip_address}:{port}"

    else:
        raise ValueError("Unsupported camera type")

