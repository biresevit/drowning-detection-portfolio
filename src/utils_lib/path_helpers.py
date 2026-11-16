import os
is_docker_env = os.path.exists("/.dockerenv")
def normalize_path(path: str) -> str:
    """
    Normalize a path depending on the operating system.

    - On Windows (os.name == 'nt'): if the path starts with '/', remove it.
    - On other systems (Linux, Docker, etc.): if the path does NOT start with '/', add it.
    """

    # Check if running in Docker
    if not is_docker_env:
        if path.startswith("/"):
            return path[1:]
    if os.name == "nt":
        # Windows – remove the leading '/' if it exists
        if path.startswith("/"):
            return path[1:]
        return path
    else:
        # Linux/Docker – add a leading '/' if it doesn't exist
        if not path.startswith("/"):
            return "/" + path
        return path
