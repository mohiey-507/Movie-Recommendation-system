import yaml
from pathlib import Path

class ConfigLoader:
    @staticmethod
    def load_config(env='local'):
        config_path = Path(__file__).parent / f"{env}_config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)