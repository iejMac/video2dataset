import os
from omegaconf import OmegaConf

configs_path = os.path.dirname(os.path.realpath(__file__))

CONFIGS = {"default": OmegaConf.load(os.path.join(configs_path, "default.yaml"))}
