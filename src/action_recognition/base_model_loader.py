import os
import logging
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from utils_lib.path_helpers import normalize_path

logger = logging.getLogger()


class BaseModelLoader:
    def __init__(self, models_dir="/models"):
        models_dir = normalize_path(models_dir)
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            logger.error(f"Models folder '{self.models_dir}' does not exist.")
        else:
            logger.info(f"Models folder '{self.models_dir}' exists. Listing contents:")
            for file_name in os.listdir(self.models_dir):
                full_path = os.path.join(self.models_dir, file_name)
                if os.path.isfile(full_path):
                    logger.info(f"- {file_name}")

    def get_model_path(self, model_name, extension):
        is_docker_env = os.path.exists("/.dockerenv")
        path = os.path.join(self.models_dir, f"{model_name}{extension}")
        if (os.name == "nt" or not is_docker_env) and path.startswith("/"):
            path = path[1:]
        return path
    
    @staticmethod
    def decrypt_model_file(enc_file: str, key) -> bytes:
        """
        Decrypts an AES-encrypted model file.
        """
        with open(enc_file, 'rb') as f:
            iv = f.read(16)
            encrypted_data = f.read()

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()

        return data


