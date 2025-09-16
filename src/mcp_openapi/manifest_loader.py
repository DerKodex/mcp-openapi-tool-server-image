import os
import yaml

DEFAULT_MANIFEST_PATH = "/config/driver-manifest.yaml"


def load_manifest(path=None):
    """
    Load and parse the driver-manifest.yaml file.
    Args:
        path (str): Path to the manifest file. Defaults to /config/driver-manifest.yaml
    Returns:
        dict: Parsed manifest as a Python dictionary.
    Raises:
        FileNotFoundError: If the manifest file does not exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    manifest_path = path or os.environ.get("MANIFEST_PATH", DEFAULT_MANIFEST_PATH)
    with open(manifest_path, "r") as f:
        return yaml.safe_load(f)
