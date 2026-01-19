import yaml
import argparse
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CropGAN Project")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args, unknown = parser.parse_known_args()
    return args
